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
    private const string ResearchModelAlias = "qwen3-0.6b";
    private const string PreferredVariantHint = "cpu";
    private static string CurrentDefaultModelAlias = ResearchModelAlias;
    private static string? CurrentDefaultVariantId;
    private static readonly JsonSerializerOptions FileJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

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
    string defaultModelPath = Path.Combine(appRoot, "default-model.json");
    ApplyPersistedDefaultSelection(defaultModelPath);
        LoadedModelSession? activeSession = null;

        try
        {
            while (true)
            {
                PrintMainMenu();

                string? option = Console.ReadLine()?.Trim();
                Console.WriteLine();

                switch (option)
                {
                    case "1":
                    {
                        activeSession = await SelectDefaultModelAsync(
                            activeSession,
                            catalog,
                            availableModels,
                            modelInfoPath,
                            defaultModelPath);
                        break;
                    }
                    case "2":
                    {
                        var nextSession = await EnsureModelSessionAsync(
                            activeSession,
                            AppMode.Research,
                            catalog,
                            availableModels,
                            modelInfoPath);

                        if (nextSession is null)
                        {
                            continue;
                        }

                        activeSession = nextSession;
                        await RunResearchWorkflowAsync(activeSession.ChatClient);
                        break;
                    }
                    case "3":
                    {
                        string? imagePath = PromptImagePath();
                        if (imagePath is null)
                        {
                            continue;
                        }

                        var nextSession = await EnsureModelSessionAsync(
                            activeSession,
                            AppMode.ImageDescription,
                            catalog,
                            availableModels,
                            modelInfoPath);

                        if (nextSession is null)
                        {
                            continue;
                        }

                        activeSession = nextSession;
                        await RunImageDescriptionWorkflowAsync(activeSession.ChatClient, imagePath);
                        break;
                    }
                    case "4":
                    case "sair":
                        return;
                    default:
                        Console.WriteLine("Opcao invalida. Escolha 1, 2, 3 ou 4.\n");
                        break;
                }
            }
        }
        finally
        {
            if (activeSession is not null)
            {
                await SafeUnloadModelAsync(activeSession.Model);
            }

            Console.WriteLine("\n✅ App finalizado. Modelo descarregado.");
        }
    }

    private static void PrintMainMenu()
    {
        Console.WriteLine("Selecione uma opcao:");
        Console.WriteLine($"Modelo default atual: {CurrentDefaultModelAlias}");
        Console.WriteLine("1) Listar modelos, baixar e definir modelo default");
        Console.WriteLine("2) Pesquisar um tema com analise critica e resumo");
        Console.WriteLine("3) Descrever uma imagem");
        Console.WriteLine("4) Sair");
        Console.Write("Opcao: ");
    }

    private static async Task<LoadedModelSession?> SelectDefaultModelAsync(
        LoadedModelSession? currentSession,
        ICatalog catalog,
        List<IModel> availableModels,
        string modelInfoPath,
        string defaultModelPath)
    {
        FoundryModelInfoFile? modelInfo = LoadModelInfo(modelInfoPath);
        var orderedModels = availableModels
            .OrderBy(GetModelLabel, StringComparer.OrdinalIgnoreCase)
            .ToList();

        Console.WriteLine("Modelos disponiveis no catalogo:");
        for (int index = 0; index < orderedModels.Count; index++)
        {
            IModel model = orderedModels[index];
            string alias = GetPropertyValue(model, "Alias") ?? GetModelLabel(model);
            FoundryModelInfo? details = FindModelInfo(model, modelInfo);
            string inputModalities = GetPropertyValue(model, "InputModalities")
                ?? details?.InputModalities
                ?? "desconhecido";
            string fileSize = details?.FileSizeMb is double sizeMb
                ? $", tamanho: {sizeMb:F0} MB"
                : string.Empty;
            string cached = details?.Cached is bool cachedValue
                ? $", cache: {(cachedValue ? "sim" : "nao")}"
                : string.Empty;
            string marker = IsCurrentDefaultModel(model) ? " [default]" : string.Empty;
            Console.WriteLine($"{index + 1}) {GetModelLabel(model)} (alias: {alias}, entrada: {inputModalities}{fileSize}{cached}){marker}");
        }

        Console.Write("Escolha o numero do modelo para baixar e usar como default (ou Enter para voltar): ");
        string? selection = Console.ReadLine()?.Trim();
        Console.WriteLine();

        if (string.IsNullOrWhiteSpace(selection))
        {
            return currentSession;
        }

        if (!int.TryParse(selection, out int selectedIndex) ||
            selectedIndex < 1 ||
            selectedIndex > orderedModels.Count)
        {
            Console.WriteLine("Selecao invalida.\n");
            return currentSession;
        }

        IModel selectedModel = orderedModels[selectedIndex - 1];
        string? selectedAlias = GetPropertyValue(selectedModel, "Alias");
        if (string.IsNullOrWhiteSpace(selectedAlias))
        {
            Console.WriteLine("Nao foi possivel identificar o alias do modelo selecionado.\n");
            return currentSession;
        }

        var resolvedModel = await ResolveModelAsync(
            catalog,
            availableModels,
            modelInfoPath,
            selectedAlias,
            PreferredVariantHint,
            explicitPreferredVariantId: GetModelVariantId(selectedModel));

        if (resolvedModel is null)
        {
            Console.WriteLine("Nao foi possivel resolver o modelo selecionado no catalogo.\n");
            return currentSession;
        }

        Console.WriteLine($"Modelo selecionado: {GetModelLabel(resolvedModel)}");
        await DownloadModelAsync(resolvedModel);

        CurrentDefaultModelAlias = selectedAlias;
        CurrentDefaultVariantId = GetModelVariantId(resolvedModel);
        await SaveDefaultSelectionAsync(defaultModelPath);
        Console.WriteLine($"Modelo default atualizado para: {GetModelLabel(resolvedModel)}\n");

        if (currentSession is not null && !SessionMatchesModel(currentSession, resolvedModel))
        {
            await SafeUnloadModelAsync(currentSession.Model);
            Console.WriteLine("🔄 Modelo ativo descarregado para aplicar o novo default.\n");
            return null;
        }

        return currentSession;
    }

    private static void ApplyPersistedDefaultSelection(string defaultModelPath)
    {
        if (!File.Exists(defaultModelPath))
        {
            return;
        }

        try
        {
            string json = File.ReadAllText(defaultModelPath);
            var selection = JsonSerializer.Deserialize<DefaultModelSelectionFile>(json, FileJsonOptions);
            if (!string.IsNullOrWhiteSpace(selection?.Alias))
            {
                CurrentDefaultModelAlias = selection.Alias;
                CurrentDefaultVariantId = selection.VariantId;
            }
        }
        catch
        {
        }
    }

    private static async Task SaveDefaultSelectionAsync(string defaultModelPath)
    {
        var selection = new DefaultModelSelectionFile
        {
            Alias = CurrentDefaultModelAlias,
            VariantId = CurrentDefaultVariantId
        };

        string json = JsonSerializer.Serialize(selection, FileJsonOptions);

        await File.WriteAllTextAsync(defaultModelPath, json);
    }

    private static async Task<LoadedModelSession?> EnsureModelSessionAsync(
        LoadedModelSession? currentSession,
        AppMode desiredMode,
        ICatalog catalog,
        List<IModel> availableModels,
        string modelInfoPath)
    {
        if (currentSession?.Mode == desiredMode)
        {
            return currentSession;
        }

        IModel? nextModel = desiredMode switch
        {
            AppMode.Research => await ResolveResearchModelAsync(catalog, availableModels, modelInfoPath),
            AppMode.ImageDescription => await ResolveVisionModelAsync(catalog, availableModels, modelInfoPath),
            _ => null
        };

        if (nextModel is null)
        {
            if (desiredMode == AppMode.ImageDescription)
            {
                PrintImageModeUnavailableMessage(availableModels);
            }

            return null;
        }

        if (currentSession is not null)
        {
            await SafeUnloadModelAsync(currentSession.Model);
            Console.WriteLine("🔄 Modelo anterior descarregado.\n");
        }

        var nextSession = await LoadModelSessionAsync(nextModel, desiredMode);
        Console.WriteLine($"✅ Modo pronto: {GetModeLabel(desiredMode)}\n");
        return nextSession;
    }

    private static async Task<LoadedModelSession> LoadModelSessionAsync(IModel model, AppMode mode)
    {
        Console.WriteLine($"Variante selecionada: {DescribeModel(model)}");
        await DownloadModelAsync(model);

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
        return new LoadedModelSession(mode, model, chatClient);
    }

    private static async Task DownloadModelAsync(IModel model)
    {
        Console.WriteLine("⬇️ Verificando/baixando modelo...");

        bool completionWritten = false;
        float lastProgress = -1f;

        await model.DownloadAsync(progress =>
        {
            if (progress >= 100f)
            {
                if (!completionWritten)
                {
                    completionWritten = true;
                    Console.WriteLine("\r✅ Download concluido!      ");
                }

                return;
            }

            if (progress > lastProgress)
            {
                lastProgress = progress;
                Console.Write($"\rDownload: {progress:F1}% ");
            }
        });

        if (!completionWritten)
        {
            Console.WriteLine("\r✅ Download concluido!      ");
        }
    }

    private static async Task RunResearchWorkflowAsync(OpenAIChatClient chatClient)
    {
        Console.WriteLine("✅ Fluxo em 3 etapas carregado (Researcher -> Critic -> Summarizer)\n");

        Console.Write("Digite o tema para pesquisar (ou Enter para voltar ao menu): ");
        var input = Console.ReadLine()?.Trim();

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.WriteLine();
            return;
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
        Console.WriteLine();
    }

    private static string? PromptImagePath()
    {
        Console.Write("Informe o path da imagem (ou Enter para voltar ao menu): ");
        string? rawPath = Console.ReadLine()?.Trim().Trim('"');

        if (string.IsNullOrWhiteSpace(rawPath))
        {
            Console.WriteLine();
            return null;
        }

        string imagePath = Path.GetFullPath(rawPath);
        if (!File.Exists(imagePath))
        {
            Console.WriteLine("Arquivo de imagem nao encontrado.\n");
            return null;
        }

        string? imageType = GetImageType(imagePath);
        if (imageType is null)
        {
            Console.WriteLine("Formato de imagem nao suportado. Use PNG, JPEG, WEBP ou GIF.\n");
            return null;
        }

        return imagePath;
    }

    private static async Task RunImageDescriptionWorkflowAsync(OpenAIChatClient chatClient, string imagePath)
    {
        string? imageType = GetImageType(imagePath);
        if (imageType is null)
        {
            Console.WriteLine("Formato de imagem nao suportado. Use PNG, JPEG, WEBP ou GIF.\n");
            return;
        }

        Console.WriteLine("\n🖼️ Descrevendo imagem...\n");

        try
        {
            string description = await DescribeImageAsync(chatClient, imagePath, imageType);
            Console.WriteLine("Descricao da imagem:");
            Console.WriteLine(string.IsNullOrWhiteSpace(description) ? "Sem resultado." : description);
            Console.WriteLine("\n" + new string('-', 80));
            Console.WriteLine();
        }
        catch (Exception ex) when (IsUnsupportedMultimodalRequest(ex))
        {
            Console.WriteLine("Nao foi possivel enviar a imagem para o modelo atual.");
            Console.WriteLine("O cliente Microsoft.AI.Foundry.Local desta execucao nao suporta serializar esse payload multimodal para este modelo/runtime.\n");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Falha ao descrever a imagem.");
            Console.WriteLine($"Detalhe: {ex.Message}\n");
        }
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

    private static async Task<string> DescribeImageAsync(OpenAIChatClient chatClient, string imagePath, string imageType)
    {
        byte[] imageBytes = await File.ReadAllBytesAsync(imagePath);

        var userContents = new List<MessageContent>
        {
            MessageContent.TextContent("Descreva esta imagem em portugues brasileiro. Cite os elementos principais, o contexto provavel e detalhes relevantes de forma objetiva."),
            MessageContent.ImageBinaryContent(imageBytes, imageType, "high")
        };

        var messages = new List<ChatMessage>
        {
            new(
                role: "system",
                content: "Voce descreve imagens com foco em clareza, objetividade e detalhes uteis em portugues brasileiro.",
                name: null,
                toolCalls: null,
                toolCallId: null),
            new(
                role: "user",
                contents: userContents,
                name: null,
                toolCalls: null,
                toolCallId: null)
        };

        var response = await chatClient.CompleteChatAsync(messages);
        return response.Choices?.FirstOrDefault()?.Message?.Content ?? string.Empty;
    }

    private static async Task<IModel?> ResolveResearchModelAsync(
        ICatalog catalog,
        List<IModel> availableModels,
        string modelInfoPath)
    {
        var model = await ResolveModelAsync(
            catalog,
            availableModels,
            modelInfoPath,
            CurrentDefaultModelAlias,
            PreferredVariantHint,
            explicitPreferredVariantId: CurrentDefaultVariantId);

        if (model is not null)
        {
            return model;
        }

        Console.WriteLine($"Modelo configurado '{CurrentDefaultModelAlias}' nao foi encontrado.");
        Console.WriteLine("Modelos disponiveis:");
        foreach (var availableModel in availableModels)
        {
            Console.WriteLine($"- {GetModelLabel(availableModel)}");
        }

        var fallbackModel = availableModels
            .FirstOrDefault(candidate => VariantMatches(candidate, preferredVariantId: null, PreferredVariantHint))
            ?? availableModels.FirstOrDefault();

        if (fallbackModel is not null)
        {
            Console.WriteLine($"\nUsando o primeiro modelo disponivel: {GetModelLabel(fallbackModel)}\n");
        }

        return fallbackModel;
    }

    private static async Task<IModel?> ResolveVisionModelAsync(
        ICatalog catalog,
        List<IModel> availableModels,
        string modelInfoPath)
    {
        var detectedVisionModel = availableModels
            .Where(SupportsImageInput)
            .OrderByDescending(candidate => VariantMatches(candidate, preferredVariantId: null, PreferredVariantHint))
            .ThenBy(candidate => GetModelLabel(candidate), StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        if (detectedVisionModel is not null)
        {
            return detectedVisionModel;
        }

        var forcedAttemptModel = await ResolveModelAsync(
            catalog,
            availableModels,
            modelInfoPath,
            CurrentDefaultModelAlias,
            PreferredVariantHint,
            explicitPreferredVariantId: CurrentDefaultVariantId);

        if (forcedAttemptModel is not null)
        {
            Console.WriteLine($"Tentando descrever imagem com '{CurrentDefaultModelAlias}', embora o catalogo o classifique como texto-only.\n");
            return forcedAttemptModel;
        }

        return null;
    }

    private static async Task<IModel?> ResolveModelAsync(
        ICatalog catalog,
        List<IModel> availableModels,
        string modelInfoPath,
        string modelAlias,
        string preferredVariantHint,
        Func<IModel, bool>? predicate = null,
        string? explicitPreferredVariantId = null)
    {
        string? preferredVariantId = explicitPreferredVariantId
            ?? GetPreferredVariantId(modelInfoPath, modelAlias, preferredVariantHint);

        var directMatch = availableModels
            .FirstOrDefault(variant =>
                string.Equals(GetPropertyValue(variant, "Alias"), modelAlias, StringComparison.OrdinalIgnoreCase) &&
                VariantMatches(variant, preferredVariantId, preferredVariantHint) &&
                (predicate is null || predicate(variant)));

        if (directMatch is not null)
        {
            return directMatch;
        }

        var model = await catalog.GetModelAsync(modelAlias);
        if (model is null)
        {
            return null;
        }

        var selectedVariant = model.Variants?
            .FirstOrDefault(variant =>
                VariantMatches(variant, preferredVariantId, preferredVariantHint) &&
                (predicate is null || predicate(variant)));

        if (selectedVariant is not null)
        {
            return selectedVariant;
        }

        return predicate is null || predicate(model)
            ? model
            : null;
    }

    private static string? GetPreferredVariantId(string modelInfoPath, string modelAlias, string preferredVariantHint)
    {
        var modelInfo = LoadModelInfo(modelInfoPath);
        return modelInfo?.Models?
            .FirstOrDefault(entry =>
                string.Equals(entry.Alias, modelAlias, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(entry.Runtime?.DeviceType, preferredVariantHint, StringComparison.OrdinalIgnoreCase))
            ?.Id;
    }

    private static string GetModeLabel(AppMode mode)
    {
        return mode switch
        {
            AppMode.Research => "Pesquisar um tema com analise critica e resumo",
            AppMode.ImageDescription => "Descrever uma imagem",
            _ => "Desconhecido"
        };
    }

    private static void PrintImageModeUnavailableMessage(IEnumerable<IModel> availableModels)
    {
        Console.WriteLine("Modo de imagem indisponivel: nenhum modelo com suporte a imagem foi encontrado no catalogo local.");
        Console.WriteLine("Modelos disponiveis no catalogo:");

        foreach (IModel model in availableModels.OrderBy(GetModelLabel, StringComparer.OrdinalIgnoreCase))
        {
            string inputModalities = GetPropertyValue(model, "InputModalities") ?? "desconhecido";
            Console.WriteLine($"- {GetModelLabel(model)} (entrada: {inputModalities})");
        }

        Console.WriteLine();
    }

    private static FoundryModelInfoFile? LoadModelInfo(string modelInfoPath)
    {
        if (!File.Exists(modelInfoPath))
        {
            return null;
        }

        try
        {
            string json = File.ReadAllText(modelInfoPath);
            return JsonSerializer.Deserialize<FoundryModelInfoFile>(json, FileJsonOptions);
        }
        catch
        {
            return null;
        }
    }

    private static FoundryModelInfo? FindModelInfo(IModel model, FoundryModelInfoFile? modelInfo)
    {
        if (modelInfo?.Models is null)
        {
            return null;
        }

        string? variantId = GetModelVariantId(model);
        if (!string.IsNullOrWhiteSpace(variantId))
        {
            var exactMatch = modelInfo.Models.FirstOrDefault(entry =>
                string.Equals(entry.Id, variantId, StringComparison.OrdinalIgnoreCase));

            if (exactMatch is not null)
            {
                return exactMatch;
            }
        }

        string? alias = GetPropertyValue(model, "Alias");
        if (string.IsNullOrWhiteSpace(alias))
        {
            return null;
        }

        return modelInfo.Models.FirstOrDefault(entry =>
            string.Equals(entry.Alias, alias, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(entry.Runtime?.DeviceType, PreferredVariantHint, StringComparison.OrdinalIgnoreCase))
            ?? modelInfo.Models.FirstOrDefault(entry =>
                string.Equals(entry.Alias, alias, StringComparison.OrdinalIgnoreCase));
    }

    private static bool IsCurrentDefaultModel(IModel model)
    {
        string? alias = GetPropertyValue(model, "Alias");
        if (!string.Equals(alias, CurrentDefaultModelAlias, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string? variantId = GetModelVariantId(model);
        return string.IsNullOrWhiteSpace(CurrentDefaultVariantId) ||
            string.Equals(variantId, CurrentDefaultVariantId, StringComparison.OrdinalIgnoreCase);
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

    private static string? GetModelVariantId(IModel model)
    {
        return GetPropertyValue(model, "Id") ?? GetPropertyValue(model, "ModelId");
    }

    private static bool SupportsImageInput(IModel model)
    {
        string modalities = GetPropertyValue(model, "InputModalities") ?? string.Empty;
        if (modalities.Contains("image", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        string description = DescribeModel(model);
        return description.Contains("vision", StringComparison.OrdinalIgnoreCase)
            || description.Contains("multimodal", StringComparison.OrdinalIgnoreCase)
            || description.Contains("-vl", StringComparison.OrdinalIgnoreCase);
    }

    private static string? GetImageType(string imagePath)
    {
        string extension = Path.GetExtension(imagePath).ToLowerInvariant();
        return extension switch
        {
            ".png" => "png",
            ".jpg" => "jpeg",
            ".jpeg" => "jpeg",
            ".webp" => "webp",
            ".gif" => "gif",
            _ => null
        };
    }

    private static async Task SafeUnloadModelAsync(IModel model)
    {
        try
        {
            await model.UnloadAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠️ Nao foi possivel descarregar o modelo: {ex.Message}");
        }
    }

    private static bool IsUnsupportedMultimodalRequest(Exception ex)
    {
        Exception? current = ex;

        while (current is not null)
        {
            if (current is NotSupportedException &&
                current.Message.Contains("MessageContent", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (current.Message.Contains("ContentCalculated", StringComparison.OrdinalIgnoreCase) ||
                current.Message.Contains("JsonTypeInfo metadata", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            current = current.InnerException;
        }

        return false;
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

    private static bool SessionMatchesModel(LoadedModelSession session, IModel model)
    {
        string? modelAlias = GetPropertyValue(model, "Alias");
        string? modelVariantId = GetModelVariantId(model);

        if (!string.Equals(session.Alias, modelAlias, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!string.IsNullOrWhiteSpace(session.VariantId) &&
            !string.IsNullOrWhiteSpace(modelVariantId) &&
            !string.Equals(session.VariantId, modelVariantId, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return true;
    }

    private enum AppMode
    {
        Research,
        ImageDescription
    }

    private sealed class LoadedModelSession
    {
        public LoadedModelSession(AppMode mode, IModel model, OpenAIChatClient chatClient)
        {
            Mode = mode;
            Model = model;
            ChatClient = chatClient;
            Alias = GetPropertyValue(model, "Alias") ?? string.Empty;
            VariantId = GetModelVariantId(model);
        }

        public AppMode Mode { get; }

        public IModel Model { get; }

        public OpenAIChatClient ChatClient { get; }

        public string Alias { get; }

        public string? VariantId { get; }
    }

    private sealed class FoundryModelInfoFile
    {
        public List<FoundryModelInfo>? Models { get; set; }
    }

    private sealed class FoundryModelInfo
    {
        public string? Alias { get; set; }
        public string? Id { get; set; }
        public bool? Cached { get; set; }
        public double? FileSizeMb { get; set; }
        public string? InputModalities { get; set; }
        public FoundryRuntimeInfo? Runtime { get; set; }
    }

    private sealed class DefaultModelSelectionFile
    {
        public string? Alias { get; set; }
        public string? VariantId { get; set; }
    }

    private sealed class FoundryRuntimeInfo
    {
        public string? DeviceType { get; set; }
    }
}
