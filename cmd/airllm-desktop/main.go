// Command airllm-desktop provides a native desktop UI for AirLLM.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"github.com/lutikk/airllm/internal/gguf"
	"github.com/lutikk/airllm/internal/hub"
	"github.com/lutikk/airllm/internal/model"
	"github.com/lutikk/airllm/internal/model/gptoss"
	"github.com/lutikk/airllm/internal/model/llama"
	"github.com/lutikk/airllm/internal/ollama"
	"github.com/lutikk/airllm/internal/tokenizer/harmony"
	"github.com/lutikk/airllm/pkg/hardware"
	"github.com/lutikk/airllm/pkg/memory"
)

const (
	defaultModel       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	defaultMemory      = "4GiB"
	defaultContext     = "2048"
	defaultTokens      = "64"
	workspaceBytes     = 256 << 20
	modelDirPreference = "modelDownloadDir"
	sourceHuggingFace  = "Hugging Face"
	sourceOllama       = "Ollama"
	modelDirectoryHelp = `Идентификатор модели Hugging Face
В поле «Модель» укажите ID в формате org/repo, например TinyLlama/TinyLlama-1.1B-Chat-v1.0. ID копируется со страницы модели на huggingface.co. Этот небольшой чат-пример подходит для первой проверки, но CPU reference всё равно работает медленно.

Модель Ollama
Выберите источник Ollama, укажите ссылку вроде tinyllama:latest или gpt-oss:20b и нажмите «Загрузить». AirLLM скачивает GGUF напрямую из Ollama registry: установленная Ollama, её daemon и API не нужны. После загрузки определяется архитектура, а поле «Модель» автоматически получает готовый локальный путь.

Поддерживаемые GGUF
Llama конвертируется в safetensors F16 с config.json и tokenizer.json; поддерживаются тензоры F32, F16, Q8_0, Q4_0, Q4_K, Q5_K и Q6_K. GPT-OSS (gptoss/gpt-oss) запускается прямо из исходного model.gguf без конвертации весов: рядом создаётся o200k tokenizer.json, а запрос передаётся в минимальном Harmony-формате. Не выбирайте 120b без достаточного диска и памяти.

Закрытые модели
Если репозиторий gated или private, сначала получите доступ на его странице. Затем создайте токен в настройках Hugging Face и задайте переменную окружения HF_TOKEN до запуска AirLLM. Не вставляйте токен в поле модели.

Необходимые файлы
В репозитории или локальной папке должны быть config.json, tokenizer.json и веса safetensors. Для разбитой модели нужны все части весов и model.safetensors.index.json, если он входит в репозиторий.

Локальная модель
Вместо HF ID можно указать полный путь к папке safetensors или к GPT-OSS .gguf. Для GGUF рядом должен находиться tokenizer.json, созданный при загрузке. В этом случае AirLLM ничего не скачивает.

Каталог загрузки и диск
Файлы из Hugging Face сохраняются в выбранный каталог и используются повторно. Модели могут занимать от нескольких гигабайтов до десятков и сотен гигабайтов: заранее проверьте размер файлов на странице репозитория и свободное место. Кнопка «Выбрать» позволяет использовать другой диск; выбор сохраняется между запусками.`
)

type runParams struct {
	model         string
	modelDir      string
	source        string
	prompt        string
	memoryLimit   uint64
	contextLength int
	maxTokens     int
}

type downloadParams struct {
	model    string
	modelDir string
	source   string
}

type desktopUI struct {
	window fyne.Window

	modelEntry     *widget.Entry
	modelDirEntry  *widget.Entry
	sourceSelect   *widget.Select
	promptEntry    *widget.Entry
	memoryEntry    *widget.Entry
	contextEntry   *widget.Entry
	tokensEntry    *widget.Entry
	hardwareText   *widget.Label
	diskSpaceText  *widget.Label
	statusText     *widget.Label
	outputText     *widget.Label
	outputScroll   *container.Scroll
	progress       *widget.ProgressBar
	runButton      *widget.Button
	downloadButton *widget.Button
	cancelButton   *widget.Button

	active bool
	cancel context.CancelFunc
}

func main() {
	application := app.NewWithID("io.github.lutikk.airllm.desktop")
	application.Settings().SetTheme(&darkTheme{})
	window := application.NewWindow("AirLLM — локальная генерация")
	newDesktopUI(window)
	window.Resize(fyne.NewSize(1280, 820))
	window.ShowAndRun()
}

func newDesktopUI(window fyne.Window) *desktopUI {
	ui := &desktopUI{window: window}
	ui.modelEntry = widget.NewEntry()
	ui.modelEntry.SetText(defaultModel)
	ui.modelEntry.SetPlaceHolder("org/repo или /путь/к/модели")

	ui.modelDirEntry = widget.NewEntry()
	ui.sourceSelect = widget.NewSelect([]string{sourceHuggingFace, sourceOllama}, func(source string) {
		if source == sourceOllama {
			ui.modelEntry.SetPlaceHolder("модель:тег, например tinyllama:latest")
			return
		}
		ui.modelEntry.SetPlaceHolder("org/repo или /путь/к/модели")
	})
	ui.sourceSelect.SetSelected(sourceHuggingFace)

	ui.promptEntry = widget.NewMultiLineEntry()
	ui.promptEntry.SetPlaceHolder("Введите запрос для модели…")
	ui.promptEntry.SetMinRowsVisible(5)

	ui.memoryEntry = widget.NewEntry()
	ui.memoryEntry.SetText(defaultMemory)
	ui.contextEntry = widget.NewEntry()
	ui.contextEntry.SetText(defaultContext)
	ui.tokensEntry = widget.NewEntry()
	ui.tokensEntry.SetText(defaultTokens)

	ui.hardwareText = sectionLabel("Определение оборудования…")
	ui.diskSpaceText = sectionLabel("Определение свободного места…")
	ui.statusText = sectionLabel("Готово. Используется медленный CPU reference; Metal пока не задействован.")
	ui.outputText = widget.NewLabel("Здесь появится результат генерации.")
	ui.outputText.Wrapping = fyne.TextWrapWord
	ui.outputScroll = container.NewVScroll(ui.outputText)
	ui.progress = widget.NewProgressBar()
	ui.progress.SetValue(0)
	ui.runButton = widget.NewButtonWithIcon("Запустить", theme.MediaPlayIcon(), ui.start)
	ui.runButton.Importance = widget.HighImportance
	ui.downloadButton = widget.NewButtonWithIcon("Загрузить", theme.DownloadIcon(), ui.startDownload)
	ui.cancelButton = widget.NewButtonWithIcon("Отмена", theme.CancelIcon(), ui.stop)
	ui.cancelButton.Disable()

	preferences := fyne.CurrentApp().Preferences()
	ui.modelDirEntry.SetText(savedModelDir(preferences))
	ui.modelDirEntry.OnChanged = func(path string) {
		preferences.SetString(modelDirPreference, strings.TrimSpace(path))
		ui.updateDiskSpace()
	}

	chooseDirButton := widget.NewButtonWithIcon("Выбрать", theme.FolderOpenIcon(), func() {
		folderDialog := dialog.NewFolderOpen(func(uri fyne.ListableURI, err error) {
			if err != nil {
				dialog.ShowError(fmt.Errorf("не получилось выбрать каталог: %w", err), window)
				return
			}
			if uri == nil {
				return
			}
			ui.modelDirEntry.SetText(uri.Path())
			preferences.SetString(modelDirPreference, uri.Path())
			ui.updateDiskSpace()
		}, window)
		folderDialog.Show()
	})
	helpButton := widget.NewButtonWithIcon("Как получить модель", theme.HelpIcon(), func() {
		dialog.ShowInformation("Как получить модель", modelDirectoryHelp, window)
	})

	directoryRow := container.NewBorder(nil, nil, nil, chooseDirButton, ui.modelDirEntry)
	directorySettings := container.NewVBox(
		sectionLabel("Каталог для загрузки моделей"),
		directoryRow,
		ui.diskSpaceText,
	)
	modelSettings := styledCard("Модель", "HF ID, Ollama ref или путь к локальной папке", container.NewVBox(
		ui.sourceSelect,
		ui.modelEntry,
		container.NewGridWithColumns(2, ui.downloadButton, helpButton),
		widget.NewSeparator(),
		directorySettings,
	))

	memoryField := container.NewVBox(sectionLabel("Лимит памяти"), ui.memoryEntry)
	contextField := container.NewVBox(sectionLabel("Контекст"), ui.contextEntry)
	tokensField := container.NewVBox(sectionLabel("Новых токенов"), ui.tokensEntry)
	generationSettings := styledCard("Параметры генерации", "", container.NewGridWithColumns(2,
		memoryField, contextField, tokensField,
	))
	hardwareCard := styledCard("Среда выполнения", "", ui.hardwareText)
	settingsScroll := container.NewVScroll(container.NewVBox(modelSettings, generationSettings, hardwareCard))
	actions := container.NewGridWithColumns(2, ui.runButton, ui.cancelButton)
	left := container.NewPadded(container.NewBorder(nil, container.NewVBox(actions, ui.statusText), nil, nil, settingsScroll))

	promptCard := styledCard("Запрос", "Текст будет передан модели после загрузки", ui.promptEntry)
	resultCard := styledCard("Результат", "Потоковый вывод токенов", container.NewBorder(nil, ui.progress, nil, nil, ui.outputScroll))
	right := container.NewPadded(container.NewBorder(promptCard, nil, nil, nil, resultCard))

	split := container.NewHSplit(left, right)
	split.SetOffset(0.335)
	window.SetContent(container.NewBorder(ui.header(), nil, nil, nil, split))

	ui.updateDiskSpace()
	go func() {
		summary := hardwareSummary(hardware.Probe())
		fyne.Do(func() { ui.hardwareText.SetText(summary) })
	}()
	return ui
}

func (ui *desktopUI) header() fyne.CanvasObject {
	background := canvas.NewRectangle(colHeaderBackground)
	title := canvas.NewText("AirLLM", colForeground)
	title.TextSize = 14
	title.TextStyle = fyne.TextStyle{Bold: true}
	mode := canvas.NewText("Локальная генерация · CPU reference", colDisabled)
	mode.TextSize = 11
	row := container.NewHBox(opticallyCentered(title), layout.NewSpacer(), opticallyCentered(mode))
	content := container.New(layout.NewCustomPaddedLayout(6, 6, 16, 16), row)
	return container.New(fixedHeightLayout{height: 44}, container.NewStack(background, content))
}

func (ui *desktopUI) updateDiskSpace() {
	path := strings.TrimSpace(ui.modelDirEntry.Text)
	if path == "" {
		ui.diskSpaceText.SetText("Укажите каталог для расчёта свободного места.")
		return
	}
	ui.diskSpaceText.SetText("Определение свободного места…")
	go func() {
		free, err := diskFreeBytes(path)
		fyne.Do(func() {
			if strings.TrimSpace(ui.modelDirEntry.Text) != path {
				return
			}
			if err != nil {
				ui.diskSpaceText.SetText("Свободное место не определено: " + err.Error())
				return
			}
			ui.diskSpaceText.SetText("Свободно на диске: " + formatByteSize(free))
		})
	}()
}

func (ui *desktopUI) start() {
	if ui.active {
		return
	}
	params, err := ui.readParams()
	if err != nil {
		ui.statusText.SetText("Ошибка: " + err.Error())
		return
	}

	ui.active = true
	ui.runButton.Disable()
	ui.downloadButton.Disable()
	ui.cancelButton.Enable()
	ui.outputText.SetText("")
	ui.outputScroll.ScrollToTop()
	ui.progress.SetValue(0)
	ui.statusText.SetText("Подготовка модели…")
	ctx, cancel := context.WithCancel(context.Background())
	ui.cancel = cancel
	go ui.execute(ctx, params)
}

func (ui *desktopUI) startDownload() {
	if ui.active {
		return
	}
	params, err := ui.readDownloadParams()
	if err != nil {
		ui.statusText.SetText("Ошибка: " + err.Error())
		return
	}

	ui.active = true
	ui.runButton.Disable()
	ui.downloadButton.Disable()
	ui.cancelButton.Enable()
	ui.progress.Min = 0
	ui.progress.Max = 1
	ui.progress.SetValue(0)
	ui.statusText.SetText("Подготовка загрузки…")
	ctx, cancel := context.WithCancel(context.Background())
	ui.cancel = cancel
	go ui.executeDownload(ctx, params)
}

func (ui *desktopUI) readDownloadParams() (downloadParams, error) {
	model := strings.TrimSpace(ui.modelEntry.Text)
	if model == "" {
		return downloadParams{}, errors.New("укажите идентификатор модели")
	}
	modelDir := strings.TrimSpace(ui.modelDirEntry.Text)
	if modelDir == "" {
		return downloadParams{}, errors.New("укажите каталог для загрузки моделей")
	}
	source := ui.sourceSelect.Selected
	if source != sourceHuggingFace && source != sourceOllama {
		return downloadParams{}, errors.New("выберите источник Hugging Face или Ollama")
	}
	// Ollama refs always have a name:tag form. Respect that explicit format even
	// if the source selector was left at its default value.
	if source == sourceHuggingFace && strings.Contains(model, ":") && !strings.Contains(model, "/") {
		source = sourceOllama
	}
	return downloadParams{model: model, modelDir: modelDir, source: source}, nil
}

func (ui *desktopUI) stop() {
	if !ui.active || ui.cancel == nil {
		return
	}
	ui.cancel()
	ui.cancelButton.Disable()
	ui.statusText.SetText("Отмена…")
}

func (ui *desktopUI) readParams() (runParams, error) {
	model := strings.TrimSpace(ui.modelEntry.Text)
	if model == "" {
		return runParams{}, errors.New("укажите репозиторий Hugging Face или путь к модели")
	}
	modelDir := strings.TrimSpace(ui.modelDirEntry.Text)
	if modelDir == "" {
		return runParams{}, errors.New("укажите каталог для загрузки моделей")
	}
	if strings.TrimSpace(ui.promptEntry.Text) == "" {
		return runParams{}, errors.New("введите непустой запрос")
	}
	memoryLimit, err := memory.ParseBytes(ui.memoryEntry.Text)
	if err != nil {
		return runParams{}, fmt.Errorf("некорректный лимит памяти: %w", err)
	}
	contextLength, err := parsePositiveInt(ui.contextEntry.Text, "длина контекста")
	if err != nil {
		return runParams{}, err
	}
	maxTokens, err := strconv.Atoi(strings.TrimSpace(ui.tokensEntry.Text))
	if err != nil || maxTokens < 0 {
		return runParams{}, errors.New("максимум токенов должен быть целым неотрицательным числом")
	}
	source := ui.sourceSelect.Selected
	if source == sourceHuggingFace && strings.Contains(model, ":") && !strings.Contains(model, "/") {
		source = sourceOllama
	}
	if source != sourceHuggingFace && source != sourceOllama {
		return runParams{}, errors.New("выберите источник Hugging Face или Ollama")
	}
	return runParams{
		model:         model,
		modelDir:      modelDir,
		source:        source,
		prompt:        ui.promptEntry.Text,
		memoryLimit:   memoryLimit,
		contextLength: contextLength,
		maxTokens:     maxTokens,
	}, nil
}

func parsePositiveInt(value, name string) (int, error) {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s должна быть целым положительным числом", name)
	}
	return parsed, nil
}

func (ui *desktopUI) execute(ctx context.Context, params runParams) {
	err := ui.generate(ctx, params)
	fyne.Do(func() {
		ui.active = false
		ui.cancel = nil
		ui.runButton.Enable()
		ui.downloadButton.Enable()
		ui.cancelButton.Disable()
		ui.progress.Refresh()
		switch {
		case errors.Is(err, context.Canceled):
			ui.statusText.SetText("Отменено.")
		case err != nil:
			ui.statusText.SetText("Ошибка: " + err.Error())
		default:
			ui.progress.SetValue(1)
			ui.statusText.SetText("Генерация завершена. Режим: CPU reference.")
		}
	})
}

func (ui *desktopUI) executeDownload(ctx context.Context, params downloadParams) {
	localDir, err := ui.download(ctx, params)
	fyne.Do(func() {
		ui.active = false
		ui.cancel = nil
		ui.runButton.Enable()
		ui.downloadButton.Enable()
		ui.cancelButton.Disable()
		switch {
		case errors.Is(err, context.Canceled):
			ui.statusText.SetText("Загрузка отменена.")
		case err != nil:
			ui.statusText.SetText("Ошибка: " + err.Error())
		default:
			ui.modelEntry.SetText(localDir)
			ui.progress.Min = 0
			ui.progress.Max = 1
			ui.progress.SetValue(1)
			ui.statusText.SetText("Модель готова: " + localDir)
		}
	})
}

func (ui *desktopUI) download(ctx context.Context, params downloadParams) (string, error) {
	switch params.source {
	case sourceHuggingFace:
		localDir, err := hub.EnsureModelWithProgress(ctx, params.model, params.modelDir, func(progress hub.Progress) {
			ui.setDownloadProgress(progress.Filename, progress.Downloaded, progress.Total)
		})
		if err != nil {
			return "", fmt.Errorf("загрузка модели %s из Hugging Face: %w", params.model, err)
		}
		return localDir, nil
	case sourceOllama:
		ui.setStatus("Проверка модели и свободного места…")
		client := &ollama.Client{}
		inspection, err := client.Inspect(ctx, params.model)
		if err != nil {
			return "", fmt.Errorf("проверка модели %s в Ollama registry: %w", params.model, err)
		}
		freeBytes, err := diskFreeBytes(params.modelDir)
		if err != nil {
			return "", fmt.Errorf("проверка свободного места для %s: %w", params.modelDir, err)
		}
		if _, err := planOllamaDownload(inspection, freeBytes); err != nil {
			return "", err
		}
		ui.setStatus("Будет скачано " + formatByteSize(uint64(inspection.ModelSize)) + ".")
		pullDir := filepath.Join(params.modelDir, "ollama", safeModelReference(params.model))
		err = client.Pull(ctx, params.model, pullDir, func(progress ollama.Progress) {
			ui.setDownloadProgress(progress.Filename, progress.Downloaded, progress.Total)
		})
		if err != nil {
			return "", fmt.Errorf("загрузка модели %s из Ollama registry: %w", params.model, err)
		}
		modelPath := filepath.Join(pullDir, "model.gguf")
		architecture, err := gguf.Architecture(modelPath)
		if err != nil {
			return "", fmt.Errorf("определение архитектуры GGUF: %w", err)
		}
		switch architecture {
		case "gptoss", "gpt-oss":
			ui.setStatus("Подготовка GPT-OSS tokenizer.json без конвертации весов…")
			if _, err := gguf.PrepareGPTOSSTokenizer(modelPath, pullDir); err != nil {
				return "", fmt.Errorf("подготовка GPT-OSS tokenizer: %w", err)
			}
			return modelPath, nil
		case "llama":
			convertedDir := filepath.Join(pullDir, "airllm")
			ui.setStatus("Конвертация GGUF в safetensors F16…")
			fyne.Do(func() {
				ui.progress.Min = 0
				ui.progress.Max = 1
				ui.progress.SetValue(0)
			})
			if err := gguf.Convert(ctx, modelPath, convertedDir); err != nil {
				return "", fmt.Errorf("конвертация GGUF: %w", err)
			}
			return convertedDir, nil
		default:
			return "", fmt.Errorf("архитектура GGUF %q не поддерживается", architecture)
		}
	default:
		return "", errors.New("неподдерживаемый источник модели")
	}
}

func (ui *desktopUI) setDownloadProgress(filename string, downloaded, total int64) {
	fyne.Do(func() {
		if total > 0 {
			ui.progress.Min = 0
			ui.progress.Max = float64(total)
			ui.progress.SetValue(float64(downloaded))
			ui.statusText.SetText(fmt.Sprintf("Загрузка %s: %.1f/%.1f МиБ…", filename, float64(downloaded)/(1<<20), float64(total)/(1<<20)))
			return
		}
		ui.statusText.SetText(fmt.Sprintf("Загрузка %s: %.1f МиБ; общий размер неизвестен…", filename, float64(downloaded)/(1<<20)))
	})
}

func safeModelReference(reference string) string {
	var result strings.Builder
	for _, r := range reference {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			result.WriteRune(r)
		default:
			result.WriteByte('_')
		}
	}
	if result.Len() == 0 {
		return "model"
	}
	return result.String()
}

func (ui *desktopUI) generate(ctx context.Context, params runParams) (resultErr error) {
	modelPath := params.model
	info, statErr := os.Stat(modelPath)
	if statErr != nil {
		var err error
		modelPath, err = ui.download(ctx, downloadParams{
			model: params.model, modelDir: params.modelDir, source: params.source,
		})
		if err != nil {
			return err
		}
	} else if !info.IsDir() && !strings.EqualFold(filepath.Ext(modelPath), ".gguf") {
		return fmt.Errorf("путь модели должен указывать на каталог или GGUF-файл: %s", modelPath)
	}
	var err error
	if err := ctx.Err(); err != nil {
		return err
	}
	nativeGPTOSS := strings.EqualFold(filepath.Ext(modelPath), ".gguf")
	ui.setStatus("Проверка лимита памяти…")
	var estimate memory.Estimate
	if nativeGPTOSS {
		estimate, err = gptoss.EstimateMemory(modelPath, params.contextLength)
	} else {
		var config llama.Config
		config, err = llama.LoadConfig(filepath.Join(modelPath, "config.json"))
		if err == nil {
			var planner llama.LayerPlanner
			planner, err = llama.NewLayerPlanner(config, 4)
			if err == nil {
				estimate = planner.Estimate(uint64(params.contextLength))
			}
		}
	}
	if err != nil {
		return err
	}
	estimate.Workspace = workspaceBytes
	plan := memory.Plan(memory.Budget{HostBytes: params.memoryLimit, DeviceBytes: params.memoryLimit}, estimate)
	if !plan.Fits {
		return fmt.Errorf(
			"недостаточно памяти для CPU reference: резидентные веса %.1f МиБ, слой и вычисления %.1f МиБ, бюджет %.1f МиБ; уменьшите контекст или увеличьте лимит",
			float64(plan.HostRequiredBytes)/(1<<20),
			float64(plan.DeviceRequiredBytes)/(1<<20),
			float64(params.memoryLimit)/(1<<20),
		)
	}

	ui.setStatus("Загрузка модели и токенизатора (CPU reference)…")
	var modelRuntime model.Runtime
	if nativeGPTOSS {
		modelRuntime, err = gptoss.Open(modelPath, params.contextLength)
	} else {
		modelRuntime, err = llama.Open(modelPath, params.contextLength)
	}
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := modelRuntime.Close(); resultErr == nil && closeErr != nil {
			resultErr = fmt.Errorf("закрытие модели: %w", closeErr)
		}
	}()

	prompt := params.prompt
	if nativeGPTOSS {
		prompt = harmony.RenderPrompt(nil, prompt)
	}
	promptTokens, err := modelRuntime.Encode(prompt, !nativeGPTOSS)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ui.setStatus(fmt.Sprintf("Генерация: 0/%d токенов (CPU reference)…", params.maxTokens))
	fyne.Do(func() {
		ui.progress.Min = 0
		ui.progress.Max = float64(max(params.maxTokens, 1))
		ui.progress.SetValue(0)
	})

	generatedTokens := make([]uint32, 0, params.maxTokens)
	decodedSoFar := ""
	generated := 0
	return modelRuntime.Generate(ctx, promptTokens, params.maxTokens, func(token uint32) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		generatedTokens = append(generatedTokens, token)
		decoded, err := modelRuntime.Decode(generatedTokens, !nativeGPTOSS)
		if err != nil {
			return err
		}
		if nativeGPTOSS {
			decoded = visibleHarmonyText(decoded)
		}
		delta := decoded
		replaceOutput := false
		if strings.HasPrefix(decoded, decodedSoFar) {
			delta = strings.TrimPrefix(decoded, decodedSoFar)
			decodedSoFar = decoded
		} else {
			// A tokenizer or Harmony parser may revise earlier output.
			decodedSoFar = decoded
			replaceOutput = true
		}
		generated++
		count := generated
		fyne.Do(func() {
			if replaceOutput {
				ui.outputText.SetText(decoded)
			} else if delta != "" {
				ui.outputText.SetText(ui.outputText.Text + delta)
			}
			ui.outputScroll.ScrollToBottom()
		})
		fyne.Do(func() {
			ui.progress.SetValue(float64(count))
			ui.statusText.SetText(fmt.Sprintf("Генерация: %d/%d токенов (CPU reference)…", count, params.maxTokens))
		})
		return nil
	})
}

func visibleHarmonyText(decoded string) string {
	parser := harmony.NewParser(harmony.Options{})
	events, err := parser.WriteText(decoded)
	if err != nil {
		return decoded
	}
	var output strings.Builder
	for _, event := range events {
		if event.Type == harmony.EventText {
			output.WriteString(event.Text)
		}
	}
	// During a long analysis phase Harmony has no user-visible channel yet.
	// Never turn successful generation into a mysteriously empty answer pane.
	if output.Len() == 0 {
		return decoded
	}
	return output.String()
}

func (ui *desktopUI) setStatus(status string) {
	fyne.Do(func() { ui.statusText.SetText(status) })
}

func defaultModelDir() (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("определение каталога кэша: %w", err)
	}
	return filepath.Join(cacheDir, "airllm", "models"), nil
}

func savedModelDir(preferences fyne.Preferences) string {
	if saved := strings.TrimSpace(preferences.String(modelDirPreference)); saved != "" {
		return saved
	}
	path, err := defaultModelDir()
	if err != nil {
		return ""
	}
	return path
}

func resolveModelDir(ctx context.Context, model, downloadDir string) (string, error) {
	info, err := os.Stat(model)
	if err == nil {
		if !info.IsDir() && !strings.EqualFold(filepath.Ext(model), ".gguf") {
			return "", fmt.Errorf("локальная модель должна быть каталогом safetensors или файлом .gguf: %s", model)
		}
		return model, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("проверка локальной модели: %w", err)
	}
	if strings.TrimSpace(downloadDir) == "" {
		downloadDir, err = defaultModelDir()
		if err != nil {
			return "", err
		}
	}
	if err := os.MkdirAll(downloadDir, 0o755); err != nil {
		return "", fmt.Errorf("создание каталога загрузки: %w", err)
	}
	modelDir, err := hub.EnsureModel(ctx, model, downloadDir)
	if err != nil {
		return "", fmt.Errorf("загрузка модели %s из Hugging Face: %w", model, err)
	}
	return modelDir, nil
}

func formatByteSize(bytes uint64) string {
	if bytes < 1<<30 {
		return fmt.Sprintf("%.1f МиБ", float64(bytes)/(1<<20))
	}
	return fmt.Sprintf("%.1f ГиБ", float64(bytes)/(1<<30))
}

func hardwareSummary(snapshot hardware.Snapshot) string {
	cpu := strings.TrimSpace(snapshot.CPU.Model)
	if cpu == "" {
		cpu = "модель не определена"
	}
	ram := "не определено"
	if snapshot.CPU.RAMBytes > 0 {
		ram = fmt.Sprintf("%.1f ГиБ", float64(snapshot.CPU.RAMBytes)/(1<<30))
	}
	gpu := "не определён"
	if snapshot.GPU != nil && strings.TrimSpace(snapshot.GPU.Name) != "" {
		gpu = snapshot.GPU.Name
	}
	return fmt.Sprintf(
		"%s/%s · %s · %d ядер · ОЗУ %s · GPU %s\nДоступно: CPU · Metal %t · CUDA %t. Сейчас работает только CPU reference (медленно).",
		snapshot.CPU.OS, snapshot.CPU.Arch, cpu, snapshot.CPU.Cores, ram, gpu,
		snapshot.Capabilities.Metal, snapshot.Capabilities.CUDA,
	)
}
