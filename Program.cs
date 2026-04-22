using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Betalgo.Ranul.OpenAI.ObjectModels.RequestModels;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging.Abstractions;

class Program
{
    private const string ModelAlias = "qwen2.5-1.5b";
    private const string PreferredVariantHint = "cpu";

    static async Task Main(string[] args)
    {
        Console.WriteLine("🚀 Iniciando Microsoft Agent Framework + Foundry Local (100% self-contained)\n");

        string appRoot = Path.Combine(Environment.CurrentDirectory, ".foundry-local");
        string logsDir = Path.Combine(appRoot, "logs");
        string modelCacheDir = Path.Combine(appRoot, "models");

        Directory.CreateDirectory(appRoot);
        Directory.CreateDirectory(logsDir);
        Directory.CreateDirectory(modelCacheDir);

        var config = new Configuration
        {
            AppName = "MultiAgentQwenWorkflow",
            AppDataDir = appRoot,
            LogsDir = logsDir,
            ModelCacheDir = modelCacheDir
        };
        await FoundryLocalManager.CreateAsync(config, NullLogger.Instance);
        var manager = FoundryLocalManager.Instance;

        var catalog = await manager.GetCatalogAsync();
        var availableModels = (await catalog.ListModelsAsync()).ToList();

        if (availableModels.Count == 0)
        {
            Console.WriteLine("Nenhum modelo apareceu no catalogo inicial. Tentando registrar execution providers...\n");
            await manager.DownloadAndRegisterEpsAsync((name, progress) =>
            {
                Console.Write($"\rEP {name}: {progress:F1}%   ");
            });
            Console.WriteLine();

            catalog = await manager.GetCatalogAsync();
            availableModels = (await catalog.ListModelsAsync()).ToList();
        }

        if (availableModels.Count == 0)
        {
            throw new Exception("Nenhum modelo disponivel no catalogo do Foundry Local apos registrar os execution providers.");
        }

        string modelInfoPath = Path.Combine(modelCacheDir, "foundry.modelinfo.json");
        var model = await ResolveModelAsync(catalog, availableModels, modelInfoPath);
        if (model is null)
        {
            Console.WriteLine($"Modelo configurado '{ModelAlias}' nao foi encontrado.");
            Console.WriteLine("Modelos disponiveis:");
            foreach (var availableModel in availableModels)
            {
                Console.WriteLine($"- {GetModelLabel(availableModel)}");
            }

            model = availableModels[0];
            Console.WriteLine($"\nUsando o primeiro modelo disponivel: {GetModelLabel(model)}\n");
        }

        Console.WriteLine($"Variante selecionada: {DescribeModel(model)}");

        Console.WriteLine("⬇️ Verificando/baixando modelo...");
        await model.DownloadAsync(progress =>
        {
            if (progress < 100f)
            {
                Console.Write($"\rDownload: {progress:F1}% ");
            }
            else
            {
                Console.WriteLine("\n✅ Download concluído!");
            }
        });

        Console.WriteLine("📦 Carregando modelo...");
        try
        {
            await model.LoadAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("\n❌ Falha ao carregar o modelo.");
            Console.WriteLine(ex.ToString());
            throw;
        }
        Console.WriteLine("✅ Modelo carregado!\n");

        var chatClient = await model.GetChatClientAsync();
        Console.WriteLine("✅ Fluxo em 3 etapas carregado (Researcher -> Critic -> Summarizer)\n");

        while (true)
        {
            Console.Write("\nDigite o tema para pesquisar (ou 'sair'): ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrWhiteSpace(input) || input.Equals("sair", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            Console.WriteLine("\n🔄 Executando fluxo...\n");

            string research = await RunStepAsync(
                chatClient,
                "Voce e um pesquisador rigoroso. Busque fatos, dados e ideias sobre o tema. Seja detalhista.",
                input);
            Console.WriteLine("Researcher:");
            Console.WriteLine(research);

            string critique = await RunStepAsync(
                chatClient,
                "Voce e um critico exigente. Analise a pesquisa, aponte falhas, contradicoes e sugira melhorias.",
                $"Tema: {input}\n\nPesquisa inicial:\n{research}");
            Console.WriteLine("\nCritic:");
            Console.WriteLine(critique);

            string finalSummary = await RunStepAsync(
                chatClient,
                "Voce e um sintetizador excelente. Produza um resumo claro, objetivo e bem estruturado em portugues brasileiro.",
                $"Tema: {input}\n\nPesquisa:\n{research}\n\nCritica e melhorias:\n{critique}");

            Console.WriteLine("\n📋 RESUMO FINAL:");
            Console.WriteLine(string.IsNullOrWhiteSpace(finalSummary) ? "Sem resultado." : finalSummary);
            Console.WriteLine("\n" + new string('-', 80));
        }

        await model.UnloadAsync();
        Console.WriteLine("\n✅ App finalizado. Modelo descarregado.");
    }

    private static async Task<string> RunStepAsync(
        OpenAIChatClient chatClient,
        string instructions,
        string input)
    {
        var messages = new List<ChatMessage>
        {
            new(role: "system", content: instructions, name: null, toolCalls: null, toolCallId: null),
            new(role: "user", content: input, name: null, toolCalls: null, toolCallId: null)
        };

        var response = await chatClient.CompleteChatAsync(messages);
        return response.Choices?.FirstOrDefault()?.Message?.Content ?? string.Empty;
    }

    private static string GetModelLabel(IModel model)
    {
        string[] preferredProperties = ["DisplayName", "Alias", "Name", "Id", "ModelId"];

        foreach (string propertyName in preferredProperties)
        {
            var value = GetPropertyValue(model, propertyName);
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return model.ToString() ?? "<modelo-sem-nome>";
    }

    private static async Task<IModel?> ResolveModelAsync(ICatalog catalog, List<IModel> availableModels, string modelInfoPath)
    {
        string? preferredVariantId = null;
        if (File.Exists(modelInfoPath))
        {
            try
            {
                string json = await File.ReadAllTextAsync(modelInfoPath);
                var modelInfo = JsonSerializer.Deserialize<FoundryModelInfoFile>(json);
                preferredVariantId = modelInfo?.Models?
                    .FirstOrDefault(entry =>
                        string.Equals(entry.Alias, ModelAlias, StringComparison.OrdinalIgnoreCase) &&
                        string.Equals(entry.Runtime?.DeviceType, PreferredVariantHint, StringComparison.OrdinalIgnoreCase))
                    ?.Id;
            }
            catch
            {
                // Fall back to best-effort variant matching below if local metadata parsing fails.
            }
        }

        var directMatch = availableModels
            .FirstOrDefault(variant =>
                VariantMatches(variant, preferredVariantId, PreferredVariantHint) &&
                string.Equals(GetPropertyValue(variant, "Alias"), ModelAlias, StringComparison.OrdinalIgnoreCase));

        if (directMatch is not null)
        {
            return directMatch;
        }

        var model = await catalog.GetModelAsync(ModelAlias);
        if (model is null)
        {
            return null;
        }

        var selectedVariant = model.Variants?
            .FirstOrDefault(variant => VariantMatches(variant, preferredVariantId, PreferredVariantHint));

        if (selectedVariant is not null)
        {
            model.SelectVariant(selectedVariant);
        }

        return model;
    }

    private static string DescribeModel(IModel model)
    {
        string[] properties = ["DisplayName", "Alias", "Name", "Id", "ModelId"];
        var parts = properties
            .Select(property => GetPropertyValue(model, property))
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.OrdinalIgnoreCase);

        return string.Join(" | ", parts);
    }

    private static string? GetPropertyValue(IModel model, string propertyName)
    {
        var type = model.GetType();
        var property = type.GetProperty(propertyName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var propertyValue = property?.GetValue(model)?.ToString();
        if (!string.IsNullOrWhiteSpace(propertyValue))
        {
            return propertyValue;
        }

        var field = type.GetField($"<{propertyName}>k__BackingField", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? type.GetField(propertyName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        return field?.GetValue(model)?.ToString();
    }

    private static bool VariantMatches(IModel variant, string? preferredVariantId, string preferredVariantHint)
    {
        string description = DescribeModel(variant);
        if (!string.IsNullOrWhiteSpace(preferredVariantId) &&
            description.Contains(preferredVariantId, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return description.Contains(preferredVariantHint, StringComparison.OrdinalIgnoreCase);
    }

    private sealed class FoundryModelInfoFile
    {
        public List<FoundryModelInfo>? Models { get; set; }
    }

    private sealed class FoundryModelInfo
    {
        public string? Alias { get; set; }
        public string? Id { get; set; }
        public FoundryRuntimeInfo? Runtime { get; set; }
    }

    private sealed class FoundryRuntimeInfo
    {
        public string? DeviceType { get; set; }
    }
}
