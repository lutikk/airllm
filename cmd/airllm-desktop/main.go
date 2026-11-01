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
	"github.com/lutikk/airllm/internal/hub"
	"github.com/lutikk/airllm/internal/model/llama"
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
	modelDirectoryHelp = `Идентификатор модели Hugging Face
В поле «Модель» укажите ID в формате org/repo, например TinyLlama/TinyLlama-1.1B-Chat-v1.0. ID копируется со страницы модели на huggingface.co. Этот небольшой чат-пример подходит для первой проверки, но CPU reference всё равно работает медленно.

Закрытые модели
Если репозиторий gated или private, сначала получите доступ на его странице. Затем создайте токен в настройках Hugging Face и задайте переменную окружения HF_TOKEN до запуска AirLLM. Не вставляйте токен в поле модели.

Необходимые файлы
В репозитории или локальной папке должны быть config.json, tokenizer.json и веса safetensors. Для разбитой модели нужны все части весов и model.safetensors.index.json, если он входит в репозиторий.

Локальная модель
Вместо HF ID можно указать полный путь к уже распакованной папке модели. В этом случае AirLLM ничего не скачивает.

Каталог загрузки и диск
Файлы из Hugging Face сохраняются в выбранный каталог и используются повторно. Модели могут занимать от нескольких гигабайтов до десятков и сотен гигабайтов: заранее проверьте размер файлов на странице репозитория и свободное место. Кнопка «Выбрать» позволяет использовать другой диск; выбор сохраняется между запусками.`
)

type runParams struct {
	model         string
	modelDir      string
	prompt        string
	memoryLimit   uint64
	contextLength int
	maxTokens     int
}

type desktopUI struct {
	window fyne.Window

	modelEntry    *widget.Entry
	modelDirEntry *widget.Entry
	promptEntry   *widget.Entry
	memoryEntry   *widget.Entry
	contextEntry  *widget.Entry
	tokensEntry   *widget.Entry
	hardwareText  *widget.Label
	diskSpaceText *widget.Label
	statusText    *widget.Label
	outputText    *widget.Label
	outputScroll  *container.Scroll
	progress      *widget.ProgressBar
	runButton     *widget.Button
	cancelButton  *widget.Button

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
	modelSettings := styledCard("Модель", "HF ID или путь к локальной папке", container.NewVBox(
		ui.modelEntry,
		helpButton,
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
	ui.cancelButton.Enable()
	ui.outputText.SetText("")
	ui.outputScroll.ScrollToTop()
	ui.progress.SetValue(0)
	ui.statusText.SetText("Подготовка модели…")
	ctx, cancel := context.WithCancel(context.Background())
	ui.cancel = cancel
	go ui.execute(ctx, params)
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
	return runParams{
		model:         model,
		modelDir:      modelDir,
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

func (ui *desktopUI) generate(ctx context.Context, params runParams) (resultErr error) {
	modelDir, err := resolveModelDir(ctx, params.model, params.modelDir)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ui.setStatus("Проверка лимита памяти…")
	config, err := llama.LoadConfig(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return err
	}
	planner, err := llama.NewLayerPlanner(config, 4)
	if err != nil {
		return err
	}
	estimate := planner.Estimate(uint64(params.contextLength))
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
	modelRuntime, err := llama.Open(modelDir, params.contextLength)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := modelRuntime.Close(); resultErr == nil && closeErr != nil {
			resultErr = fmt.Errorf("закрытие модели: %w", closeErr)
		}
	}()

	promptTokens, err := modelRuntime.Encode(params.prompt, true)
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
		decoded, err := modelRuntime.Decode(generatedTokens, true)
		if err != nil {
			return err
		}
		delta := decoded
		replaceOutput := false
		if strings.HasPrefix(decoded, decodedSoFar) {
			delta = strings.TrimPrefix(decoded, decodedSoFar)
			decodedSoFar = decoded
		} else {
			// A tokenizer may revise an earlier byte sequence. Re-render the complete
			// answer instead of appending a malformed suffix.
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
		if !info.IsDir() {
			return "", fmt.Errorf("путь модели должен указывать на каталог: %s", model)
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
