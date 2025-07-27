import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import erf
import os


class XRayLensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование МПЛ")
        self.root.geometry("1200x800")

        # Параметры по умолчанию
        self.params = {
            "E_keV": 30,           # Энергия фотонов (кэВ)
            "FOCAL": 8000,         # Фокусное расстояние (мм)
            "THETA": 130,          # Угол рассеяния (градусы)
            "LENGTH": 100,         # Длина образца (мм)
            "I0": 1.0,             # Интенсивность излучения (доля)
            "rho": 2.7,            # Плотность (г/см³)
            "Z": 13,               # Атомный номер
            "A": 26.9,             # Атомная масса (г/моль)
            "angle": 0.5,          # Угол раскрытия МПЛ (градусы)
            "file_path": ""        # Путь к файлу с данными
        }

        # Физические константы
        self.h = 6.626e-34         # Постоянная Планка (Дж·с)
        self.c = 3e8              # Скорость света (м/с)
        self.e = 1.6e-19           # Заряд электрона (Кл)
        self.N_A = 6.022e23        # Число Авогадро (атомов/моль)
        self.PI = np.pi           # Число π

        self.create_widgets()
        self.setup_plots()

    def create_widgets(self):
        # Основные фреймы
        control_frame = ttk.LabelFrame(self.root, text="Параметры", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Поля ввода параметров
        ttk.Label(control_frame, text="Энергия (кэВ):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.e_kev_entry = ttk.Entry(control_frame)
        self.e_kev_entry.insert(0, str(self.params["E_keV"]))
        self.e_kev_entry.grid(row=0, column=1, pady=2)

        ttk.Label(control_frame, text="Фокусное расстояние (мм):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.focal_entry = ttk.Entry(control_frame)
        self.focal_entry.insert(0, str(self.params["FOCAL"]))
        self.focal_entry.grid(row=1, column=1, pady=2)

        ttk.Label(control_frame, text="Угол рассеяния (°):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.theta_entry = ttk.Entry(control_frame)
        self.theta_entry.insert(0, str(self.params["THETA"]))
        self.theta_entry.grid(row=2, column=1, pady=2)

        ttk.Label(control_frame, text="Длина образца (мм):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.length_entry = ttk.Entry(control_frame)
        self.length_entry.insert(0, str(self.params["LENGTH"]))
        self.length_entry.grid(row=3, column=1, pady=2)

        ttk.Label(control_frame, text="Плотность (г/см³):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.rho_entry = ttk.Entry(control_frame)
        self.rho_entry.insert(0, str(self.params["rho"]))
        self.rho_entry.grid(row=4, column=1, pady=2)

        ttk.Label(control_frame, text="Атомный номер:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.z_entry = ttk.Entry(control_frame)
        self.z_entry.insert(0, str(self.params["Z"]))
        self.z_entry.grid(row=5, column=1, pady=2)

        ttk.Label(control_frame, text="Атомная масса (г/моль):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.a_entry = ttk.Entry(control_frame)
        self.a_entry.insert(0, str(self.params["A"]))
        self.a_entry.grid(row=6, column=1, pady=2)

        ttk.Label(control_frame, text="Угол раскрытия (°):").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.angle_entry = ttk.Entry(control_frame)
        self.angle_entry.insert(0, str(self.params["angle"]))
        self.angle_entry.grid(row=7, column=1, pady=2)

        # Кнопка выбора файла
        ttk.Button(control_frame, text="Выбрать файл с данными", command=self.select_file).grid(row=8, column=0, columnspan=2, pady=10)
        self.file_label = ttk.Label(control_frame, text="Файл не выбран", wraplength=200)
        self.file_label.grid(row=9, column=0, columnspan=2, pady=5)

        # Кнопки управления
        ttk.Button(control_frame, text="Рассчитать", command=self.run_calculation).grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Сохранить графики", command=self.save_plots).grid(row=11, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Справка", command=self.show_help).grid(row=12, column=0, columnspan=2, pady=5)

        # Вкладки для графиков
        self.notebook = ttk.Notebook(self.plot_frame)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        self.tab1 = ttk.Frame(self.notebook)  # Профиль МПЛ
        self.tab2 = ttk.Frame(self.notebook)  # Функция толщины
        self.tab3 = ttk.Frame(self.notebook)  # Интенсивность
        self.tab4 = ttk.Frame(self.notebook)  # Параметры

        self.notebook.add(self.tab1, text="Профиль МПЛ")
        self.notebook.add(self.tab2, text="Функция толщины")
        self.notebook.add(self.tab3, text="Интенсивность")
        self.notebook.add(self.tab4, text="Параметры")

        # Фрейм для кнопок управления на вкладке "Функция толщины"
        self.thickness_control_frame = ttk.Frame(self.tab2)
        self.thickness_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Кнопки управления
        self.show_real_btn = ttk.Button(
            self.thickness_control_frame,
            text="Реальный профиль",
            command=lambda: self.show_thickness_profile("real")
        )
        self.show_real_btn.pack(side=tk.LEFT, padx=5)

        self.show_theory_btn = ttk.Button(
            self.thickness_control_frame,
            text="Теоретический профиль",
            command=lambda: self.show_thickness_profile("theory")
        )
        self.show_theory_btn.pack(side=tk.LEFT, padx=5)

        self.show_both_btn = ttk.Button(
            self.thickness_control_frame,
            text="Сравнение",
            command=lambda: self.show_thickness_profile("both")
        )
        self.show_both_btn.pack(side=tk.LEFT, padx=5)

        # Создаем два отдельных фрейма для управления на вкладке интенсивности
        self.profile_control_frame = ttk.Frame(self.tab3)
        self.profile_control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_control_frame = ttk.Frame(self.tab3)
        self.mode_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Кнопки выбора профиля (верхний ряд)
        self.profile_mode = tk.StringVar(value="both")  # По умолчанию показываем оба
        ttk.Label(self.profile_control_frame, text="Профиль:").pack(side=tk.LEFT, padx=5)

        self.show_real_btn = ttk.Radiobutton(
            self.profile_control_frame,
            text="Реальный",
            variable=self.profile_mode,
            value="real",
            command=self.update_intensity_plot
        )
        self.show_real_btn.pack(side=tk.LEFT, padx=5)

        self.show_theory_btn = ttk.Radiobutton(
            self.profile_control_frame,
            text="Теоретический",
            variable=self.profile_mode,
            value="theory",
            command=self.update_intensity_plot
        )
        self.show_theory_btn.pack(side=tk.LEFT, padx=5)

        self.show_both_btn = ttk.Radiobutton(
            self.profile_control_frame,
            text="Сравнение",
            variable=self.profile_mode,
            value="both",
            command=self.update_intensity_plot
        )
        self.show_both_btn.pack(side=tk.LEFT, padx=5)

        # Переключатели амплитуда/фаза (нижний ряд)
        self.data_mode = tk.StringVar(value="amplitude")  # По умолчанию амплитуда
        ttk.Label(self.mode_control_frame, text="Отображение:").pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            self.mode_control_frame,
            text="Амплитуда",
            variable=self.data_mode,
            value="amplitude",
            command=self.update_intensity_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            self.mode_control_frame,
            text="Фаза",
            variable=self.data_mode,
            value="phase",
            command=self.update_intensity_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            self.mode_control_frame,
            text="Сравнение",
            variable=self.data_mode,
            value="both",
            command=self.update_intensity_plot
        ).pack(side=tk.LEFT, padx=5)

        # Текстовое поле для вывода параметров
        self.text_output = tk.Text(self.tab4, wrap=tk.WORD)
        self.text_output.pack(expand=True, fill=tk.BOTH)

    def update_intensity_plot(self):
        """Обновляет график с учетом выбранных профиля и режима"""
        self.ax3_amp.clear()
        self.ax3_phase.clear()

        profile = self.profile_mode.get()
        mode = self.data_mode.get()

        # Подготавливаем данные
        if profile in ["theory", "both"]:
            I_theoretical = np.abs(self.fresnel_diffraction_field)
            phase_theoretical = np.angle(self.output_wave)

        if profile in ["real", "both"]:
            I_real = np.abs(self.fresnel_diffraction_field_real)
            phase_real = np.angle(self.output_wave_real)

        # Отрисовываем амплитуду
        if mode in ["amplitude", "both"]:
            if profile in ["theory", "both"]:
                self.ax3_amp.plot(
                    self.x_new - 0.3, I_theoretical,
                    color='blue', linewidth=2,
                    label='Теоретическая интенсивность'
                )
            if profile in ["real", "both"]:
                self.ax3_amp.plot(
                    self.x_new - 0.3, I_real,
                    color='red', linewidth=2,
                    label='Реальная интенсивность'
                )
            self.ax3_amp.set_ylabel('Интенсивность излучения (у.е.)', color='blue')
            self.ax3_amp.tick_params(axis='y', labelcolor='blue')

        # Отрисовываем фазу
        if mode in ["phase", "both"]:
            if profile in ["theory", "both"]:
                self.ax3_phase.plot(
                    self.x_new - 0.3, phase_theoretical,
                    color='green', linestyle='--', linewidth=2,
                    label='Теоретическая фаза'
                )
            if profile in ["real", "both"]:
                self.ax3_phase.plot(
                    self.x_new - 0.3, phase_real,
                    color='purple', linestyle='--', linewidth=2,
                    label='Реальная фаза'
                )
            self.ax3_phase.set_ylabel('Фаза (рад)', color='green')
            self.ax3_phase.tick_params(axis='y', labelcolor='green')

        # Общие настройки
        self.ax3_amp.set_title('Амплитуда и фаза волны, прошедшей оптический элемент')
        self.ax3_amp.set_xlabel('Расстояние от оптической оси (мм)')
        self.ax3_amp.grid(True)

        # Объединяем легенды
        lines, labels = [], []
        if mode in ["amplitude", "both"]:
            l, lab = self.ax3_amp.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        if mode in ["phase", "both"]:
            l, lab = self.ax3_phase.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)

        if lines:  # Если есть что показывать в легенде
            self.ax3_amp.legend(lines, labels, loc='upper right')

        self.ax3_amp.set_xlim(left=-0.5, right=0.5)
        self.ax3_amp.set_ylim(bottom=0)

        self.canvas3.draw()

    def show_thickness_profile(self, mode):
        """Показывает выбранный профиль толщины"""
        self.ax2.clear()

        if mode == "real" or mode == "both":
            self.ax2.plot(
                self.x_new - 0.3,
                self.our_lens_thickness,
                color='red',
                linewidth=2,
                label='Реальный профиль'
            )

        if mode == "theory" or mode == "both":
            self.ax2.plot(
                self.x_new - 0.3,
                self.lens_thickness,
                color='blue',
                linewidth=2,
                linestyle='-',
                label='Теоретический профиль'
            )

        self.ax2.set_xlabel('Расстояние от оптической оси (мм)')
        self.ax2.set_ylabel('Толщина МПЛ (мм)')
        self.ax2.set_title('Функция толщины МПЛ')
        self.ax2.grid(True, linestyle=':', alpha=0.7)

        if mode == "both":
            self.ax2.legend(fontsize=10)

        self.ax2.set_xlim(left=-1, right=1)
        self.ax2.set_ylim(bottom=0)

        self.canvas2.draw()

    def setup_plots(self):
        # Инициализация графиков
        self.fig1, self.ax1 = plt.subplots(figsize=(8, 5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # Создаем второй y-axes для фазы
        self.fig3, self.ax3_amp = plt.subplots(figsize=(8, 5))
        self.ax3_phase = self.ax3_amp.twinx()  # Создаем вторую ось для фазы
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # Заглушки для графиков
        self.ax1.text(0.5, 0.5, "Данные не загружены", ha='center', va='center')
        self.ax2.text(0.5, 0.5, "Данные не загружены", ha='center', va='center')
        self.ax3_amp.text(0.5, 0.5, "Данные не загружены", ha='center', va='center')

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.params["file_path"] = file_path
            self.file_label.config(text=os.path.basename(file_path))

    def get_input_params(self):
        try:
            self.params["E_keV"] = float(self.e_kev_entry.get())
            self.params["FOCAL"] = float(self.focal_entry.get())
            self.params["THETA"] = float(self.theta_entry.get())
            self.params["LENGTH"] = float(self.length_entry.get())
            self.params["rho"] = float(self.rho_entry.get())
            self.params["Z"] = int(self.z_entry.get())
            self.params["A"] = float(self.a_entry.get())
            self.params["angle"] = float(self.angle_entry.get())

            if not self.params["file_path"]:
                messagebox.showerror("Ошибка", "Не выбран файл с данными")
                return False

            return True
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {str(e)}")
            return False

    def run_calculation(self):
        if not self.get_input_params():
            return

        try:
            # Обновляем физические параметры
            self.update_physical_params()

            # Выполняем расчеты
            self.calculate()

            # Обновляем графики
            self.update_plots()

            # Выводим параметры
            self.show_parameters()

        except Exception as e:
            messagebox.showerror("Ошибка расчета", f"Произошла ошибка: {str(e)}")

    def update_physical_params(self):
        # Вычисляемые величины
        self.LAMBDA = self.h * self.c / (self.params["E_keV"] * self.e)  # Длина волны (м), E_keV -> эВ
        self.K = 2 * self.PI / self.LAMBDA                  # Волновое число (м⁻¹)
        self.DELTA = 2.7e8 * self.params["rho"] * self.params["Z"] / self.params["A"] * self.LAMBDA**2  # Поправка на преломление

    def calculate(self):
        # Здесь переносим основные расчетные функции из оригинального кода
        self.res = 10000
        self.xmin, self.xmax = -0.3, 0.3
        self.source_lens_distance = 2 * self.params["FOCAL"]

        # Чтение данных
        self.x, self.y = self.read_coordinates(self.params["file_path"])
        self.x_rotated, self.y_rotated = self.rotate_graphic(self.params["angle"], self.x, self.y)

        # Моделирование лучей
        self.xs = np.linspace(self.xmin, self.xmax, self.res)
        self.dx = (self.xmax - self.xmin) / (self.res - 1)
        self.fx = np.fft.fftfreq(self.res, d=self.dx)
        self.fx = np.fft.fftshift(self.fx)

        self.list_paths = self.simulate_rays(self.xs, self.x_rotated, self.y_rotated)

        # Собираем всю числовую прямую
        self.xs_combined = np.concatenate((-self.xs + 0.6, self.xs))
        self.list_paths_combined = np.concatenate((np.max(self.list_paths) - self.list_paths, np.max(self.list_paths) - self.list_paths))

        # Сортируем по возрастанию
        sorted_indices = np.argsort(self.xs_combined)
        self.xs_combined_sorted = self.xs_combined[sorted_indices]
        self.list_paths_combined_sorted = self.list_paths_combined[sorted_indices]

        # Оцениваем радиусы
        self.radii_lens = self.analyze_lens_structure(self.x, self.y)

        # Расчет параметров МПЛ
        self.params_lens = self.calculate_focal_properties(self.x, self.y, self.x_rotated, self.y_rotated)

        # Теоретическая линза
        self.x_new = np.linspace(np.min(self.xs_combined_sorted), np.max(self.xs_combined_sorted), self.res)
        self.lens_thickness = self.perfect_line(self.x_new, self.params["FOCAL"], offset=0.3)

        # Реальная МПЛ
        self.our_lens_thickness = np.interp(self.x_new, self.xs_combined_sorted, self.list_paths_combined_sorted)

        # Рассчет фокуса реальной МПЛ
        self.focal_length_real = self.fit_perfect_line(self.xs_combined_sorted, self.list_paths_combined_sorted, offset=0.3)

        # Амплитуда и фаза волны, прошедшей оптический элемент
        self.output_wave = self.spherical_wave(self.x_new, self.source_lens_distance) * self.transmission_function(self.lens_thickness)
        self.output_wave_real = self.spherical_wave(self.x_new, self.source_lens_distance) * self.transmission_function(self.our_lens_thickness)
        # Амплитуда и фаза волнового фронта на экране
        self.x_screen = self.fx * self.LAMBDA * self.source_lens_distance
        self.fresnel_diffraction_field = self.fresnel_diffraction(self.output_wave, self.x_new, self.source_lens_distance, self.x_screen)
        self.fresnel_diffraction_field_real = self.fresnel_diffraction(self.output_wave_real, self.x_new, self.source_lens_distance, self.x_screen)

        #plt.plot(self.x_screen, np.abs(self.fresnel_diffraction_field), color='blue', linewidth=2, linestyle='-', label='Теоретическая интенсивность излучения')
        #plt.plot(self.x_screen, np.abs(self.fresnel_diffraction_field_real), color='red', linewidth=2, linestyle='-', label='Экспериментальная интенсивность излучения')


    def update_plots(self):
        # Очищаем графики
        self.ax1.clear()
        self.ax2.clear()
        self.ax3_amp.clear()

        # График профиля МПЛ
        self.ax1.plot(self.x, self.y, color="blue")
        self.ax1.plot(self.x_rotated, self.y_rotated, color="red")

        line1 = Line2D([0], [0], color="blue", linestyle="-", linewidth=2, label="Нижний профиль МПЛ")
        line2 = Line2D([0], [0], color="red", linestyle="-", linewidth=2, label=f"Нижний профиль МПЛ с углом раскрытия {self.params['angle']}°")

        self.ax1.axis("equal")
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, np.max(self.x) / 10)
        self.ax1.legend(handles=[line1, line2])
        self.ax1.grid(True)
        self.ax1.set_title("Профиль МПЛ")

        # По умолчанию показываем оба профиля толщины
        self.show_thickness_profile("both")

        # График интенсивности
        #self.ax3_amp.plot(self.x_new - 0.3, np.abs(self.output_wave), linewidth=2, label=f'Теоретическая, {self.params["E_keV"]} кэВ', color="blue")
        #self.ax3_amp.plot(self.x_new - 0.3, np.abs(self.output_wave_real), linewidth=2, label=f'Реальная, {self.params["E_keV"]} кэВ', color="red")
        self.ax3_amp.plot(self.x_screen, np.abs(self.fresnel_diffraction_field), color='blue', linewidth=2, linestyle='-', label='Теоретическая интенсивность излучения')
        self.ax3_amp.plot(self.x_screen, np.abs(self.fresnel_diffraction_field_real), color='red', linewidth=2, linestyle='-', label='Экспериментальная интенсивность излучения')

        self.ax3_amp.set_xlim(-0.5, 0.5)
        self.ax3_amp.set_xlabel('Расстояние от оптической оси (мм)')
        self.ax3_amp.set_ylabel('Интенсивность излучения (у.е.)')
        self.ax3_amp.set_title('Амплитуда и фаза волны, прошедшей оптический элемент')
        self.ax3_amp.grid(True, linestyle='-', alpha=0.7)
        self.ax3_amp.legend(fontsize=10)

        # Обновляем canvas
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        # Обновляем график интенсивности с учетом текущего режима
        self.update_intensity_plot()

    def show_parameters(self):
        # Собираем все данные
        params = [
            ("Длина МПЛ", f"{self.params['LENGTH']} мм"),
            ("Высота зуба y_t", f"{self.params_lens['y_t']:.4f} мм"),
            ("Высота раскрытия y_g", f"{self.params_lens['y_g']:.4f} мм"),
            ("Внутренний радиус", f"{int(self.radii_lens['Inner Radius'])} μm"),
            ("Внешний радиус", f"{int(np.abs(self.radii_lens['Outer Radius']))} μm"),
            ("Длина волны", f"{self.LAMBDA:.4e} м"),
            ("Фокусное расстояние (теория)", f"{self.params['FOCAL']:.2f} мм"),
            ("Фокусное расстояние (реальность)", f"{self.focal_length_real:.2f} мм"),
            ("σ_abs (теория)", f"{self.absorption_aperture(self.params['FOCAL']):.4f} μm"),
            ("σ_abs (реальность)", f"{self.absorption_aperture(self.focal_length_real):.4f} μm"),
            ("μ (коэф. поглощения)", f"{self.linear_absorption_coefficient():.6f} см^-1"),
            ("β (мнимая часть показателя преломления)", f"{self.beta():.4e}"),
            ("δ (декремент преломления)", f"{self.DELTA:.4e}"),
            ("T_avg (теория)", f"{self.Tabg(self.params['FOCAL']):.6f}"),
            ("T_avg (реальность)", f"{self.Tabg(self.focal_length_real):.6f}")
        ]

        # Форматируем вывод
        output = "Результаты расчета:\n\n"
        for name, value in params:
            output += f"{name}: {value}\n"

        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, output)

    def save_plots(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            try:
                self.fig1.savefig(os.path.join(dir_path, "lens_profile.png"))
                self.fig2.savefig(os.path.join(dir_path, "thickness_comparison.png"))
                self.fig3.savefig(os.path.join(dir_path, "intensity.png"))

                # Сохраняем параметры в файл
                with open(os.path.join(dir_path, "parameters.txt"), "w") as f:
                    f.write(self.text_output.get(1.0, tk.END))

                messagebox.showinfo("Сохранение", "Графики и параметры успешно сохранены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файлы: {str(e)}")

    def show_help(self):
        try:
            with open('help.txt', 'r', encoding='utf-8') as f:
                help_text = f.read()

            help_window = tk.Toplevel(self.root)
            help_window.title("Руководство пользователя")

            text = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
            text.pack(expand=True, fill=tk.BOTH)
            text.insert(tk.END, help_text)
            text.config(state=tk.DISABLED)

            scroll = ttk.Scrollbar(help_window, command=text.yview)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            text.config(yscrollcommand=scroll.set)

        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Файл справки не найден")

    # Далее идут методы из оригинального кода, которые нужно перенести:
    def read_coordinates(self, file_path):
        """Читает координаты из CSV-файла"""
        coordinates = pd.read_csv(file_path, header=None, sep=',', dtype={"column_name": float})
        x = coordinates[0].values
        y = coordinates[1].values * 1e-3  # Переводим в метры
        return x, y

    def rotate_graphic(self, angle_deg, x, y):
        """Поворачивает профиль на заданный угол"""
        angle = -np.radians(angle_deg)
        pivot = np.array([x[0], y[0]], dtype=np.float64)

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=np.float64)

        translated_coordinates = np.vstack((x, y)) - pivot[:, np.newaxis]
        rotated_coordinates = rotation_matrix @ translated_coordinates
        rotated_coordinates += pivot[:, np.newaxis]

        return rotated_coordinates[0, :], rotated_coordinates[1, :]

    def simulate_rays(self, xs, x_rotated, y_rotated):
        """Моделирует прохождение лучей через МПЛ"""
        rays = []
        for j in range(len(xs)):
            rays.append(
                self.covered_in_the_material(0, xs[j], self.params["LENGTH"], x_rotated, y_rotated, self.res)
            )
        return rays

    def covered_in_the_material(self, x_start, y_start, length, x_teeth, y_teeth, num_points):
        """Рассчитывает пройденный путь в материале"""
        x_coords = np.linspace(x_start, length, num_points)
        y_coords = 0 * x_coords + y_start

        y_teeth_interp = np.interp(x_coords, x_teeth, y_teeth)

        path = 0.0
        is_inside = False
        start_inside = -1

        for i in range(len(x_coords)):
            if y_coords[i] >= y_teeth_interp[i]:
                if not is_inside:
                    is_inside = True
                    start_inside = i
            else:
                if is_inside:
                    is_inside = False
                    end_inside = i
                    path += np.sqrt(
                        (x_coords[end_inside] - x_coords[start_inside]) ** 2
                        + (y_coords[end_inside] - y_coords[start_inside]) ** 2
                    )

        if is_inside:
            end_inside = len(x_coords) - 1
            path += np.sqrt(
                (x_coords[end_inside] - x_coords[start_inside]) ** 2
                + (y_coords[end_inside] - y_coords[start_inside]) ** 2
            )

        return path

    def analyze_lens_structure(self, x, y, period=1):
        """Анализирует радиусы МПЛ"""
        half_period = period / 2

        # Внутренний радиус
        start_index_inner = self.find_first_close_enough(x, 5 * half_period - half_period / 2)
        end_index_inner = self.find_first_close_enough(x, 5 * half_period + half_period / 2)
        radius_inner = self.estimate_radii(x, y, start_index_inner, end_index_inner) * 1e3

        # Внешний радиус
        start_index_outer = self.find_first_close_enough(x, 4 * half_period - half_period / 2)
        end_index_outer = self.find_first_close_enough(x, 4 * half_period + half_period / 2)
        radius_outer = self.estimate_radii(x, y, start_index_outer, end_index_outer) * 1e3

        return {
            "Inner Radius": radius_inner,
            "Outer Radius": radius_outer
        }

    def find_first_close_enough(self, data, value):
        """Находит первый элемент в массиве, больший или равный value"""
        for d in data:
            if d >= value:
                return list(data).index(d)
        return None

    def estimate_radii(self, x_data, y_data, start_index, end_index):
        """Оценивает радиусы по сегменту данных"""
        x_segment = x_data[start_index:end_index]
        y_segment = y_data[start_index:end_index]

        x_center_guess = np.mean(x_segment)
        y_center_guess = np.max(y_segment)
        radius_guess = np.ptp(x_segment)

        initial_guess = [x_center_guess, y_center_guess, radius_guess, np.pi / 6, np.pi / 6]

        try:
            popt, _ = curve_fit(self.teeth_curve, x_segment, y_segment, p0=initial_guess,
                               bounds=([0, min(x_data), min(y_data), 0, 0],
                               [np.max(x_data) - np.min(x_data), max(x_data), max(y_data) * 2, np.pi / 2, np.pi / 2]),
                               method='trf')
            return popt[2]
        except RuntimeError:
            return 0

    def teeth_curve(self, x, xr, yr, r, theta_start, theta_end):
        """Функция формы зуба МПЛ"""
        y = np.zeros_like(x, dtype=float)

        one = x < xr - r * np.cos(theta_start)
        y[one] = yr + r * np.sin(theta_start) - (xr - r * np.cos(theta_start) - x[one]) / np.tan(theta_start)

        two = (x >= xr - r * np.cos(theta_start)) & (x <= xr + r * np.cos(theta_end))
        y[two] = np.sqrt(r ** 2 - (x[two] - xr) ** 2) + yr

        three = x > xr + r * np.cos(theta_end)
        y[three] = yr + r * np.sin(theta_end) - (x[three] - xr - r * np.cos(theta_end)) / np.tan(theta_end)

        return -y

    def perfect_line(self, x, fdist, offset=0.0):
        """Идеальная функция толщины линзы"""
        return (x - offset) ** 2 / (2 * fdist * self.DELTA)

    def fit_perfect_line(self, x_data, y_data, offset=0.0):
        """Подгоняет идеальную функцию толщины"""
        initial_guess = [8000.0]

        try:
            popt, _ = curve_fit(lambda x, fdist: self.perfect_line(x, fdist, offset), x_data, y_data, p0=initial_guess)
            return popt[0]
        except RuntimeError:
            return 0

    def calculate_focal_properties(self, x, y, x_rotated, y_rotated, period=1):
        """Рассчитывает фокусные характеристики МПЛ"""

        x_lim = np.max(x)
        half_period = period / 2
        list_min = []
        list_max = []

        for i in range(1, int(x_lim * period) + 1):
            array_for_min = y[self.find_first_close_enough(x, i * half_period - half_period / 2):
                            self.find_first_close_enough(x, i * half_period + half_period / 2)]
            array_for_max = y[self.find_first_close_enough(x, i * 2 * half_period - half_period / 2):
                            self.find_first_close_enough(x, i * 2 * half_period + half_period / 2)]
            list_min.append(np.min(array_for_min))
            list_max.append(np.max(array_for_max))

        average_min = np.mean(list_min)
        average_max = np.mean(list_max)
        y_t = np.abs(average_min) + np.abs(average_max)

        first_peak = y_rotated[self.find_first_close_enough(x_rotated, 0):
                     self.find_first_close_enough(x_rotated, half_period)]
        last_peak = y_rotated[self.find_first_close_enough(x_rotated, 97 * 2 * half_period - half_period / 2):
                    self.find_first_close_enough(x_rotated, 97 * 2 * half_period + half_period / 2)]
        y_g = np.abs(np.max(first_peak)) + np.abs(np.max(last_peak))

        return {
            'y_t': y_t,
            'y_g': y_g
        }

    def beta(self):
        """Вычисляет мнимую часть показателя преломления"""
        return self.linear_absorption_coefficient() * self.LAMBDA / (4 * self.PI)

    def linear_absorption_coefficient(self):
        """Вычисляет линейный коэффициент поглощения"""
        sigma = 24.2 * (self.params["Z"] ** 4.2) * (self.params["E_keV"] ** -3) + 0.56 * self.params["Z"]
        sigma *= 1e-24  # в см²

        n_a = self.N_A * self.params["rho"] / self.params["A"]  # атомов/см³
        return n_a * sigma  # см^-1

    def attenuation_length(self):
        """Вычисляет длину поглощения"""
        return 1 / self.linear_absorption_coefficient()

    def absorption_aperture(self, F):
        """Вычисляет среднеквадратичную ширину пучка из-за поглощения"""
        return np.sqrt(F * self.DELTA * self.attenuation_length())

    def transmission_profile(self, b, F):
        """Вычисляет профиль пропускания для пучка шириной b"""
        return np.sqrt(2 * np.pi) * (self.absorption_aperture(F) / b) * erf(b / (2 * np.sqrt(2) * self.absorption_aperture(F)))

    def phase_error_from_roughness(self, R_q, N, theta_deg):
        """Вычисляет фазовую ошибку из-за шероховатости поверхности"""
        theta_rad = np.deg2rad(theta_deg)
        return (2 * np.pi * R_q * np.sqrt(2 * N * self.DELTA)) / (self.LAMBDA * np.sin(theta_rad))

    def intensity_reduction_from_roughness(self, sigma_phi_max, sigma_abs, y_g):
        """Вычисляет уменьшение интенсивности из-за шероховатости"""
        epsilon = (2 / np.sqrt(np.pi)) * (sigma_abs / y_g)
        return np.exp(-epsilon * (sigma_phi_max ** 2))

    def transmitted_intensity(self, thickness):
        """Вычисляет интенсивность после прохождения МПЛ"""
        return self.params["I0"] * np.exp(-self.linear_absorption_coefficient() * thickness)

    def transmission_function(self, thickness):
        """Вычисляет комплексную функцию пропускания"""
        return np.exp(-1j * self.K * (self.DELTA - 1j * self.beta()) * thickness)

    def spherical_wave(self, x, y):
        """Вычисляет комплексную амплитуду сферической волны"""
        return self.params["E_keV"] * np.exp(1j * self.K * np.sqrt(x ** 2 + y ** 2)) / np.sqrt(x ** 2 + y ** 2)

    def fresnel_diffraction(self, output_wave, x0, z, x):
        """Вычисляет поле на экране в приближении Френеля"""
        fresnel_phase = np.exp(self.PI * 1j / (self.LAMBDA * z) * (x0 ** 2))

        fresnel_propagator = (
            -1j
            / self.LAMBDA
            * np.exp(
                2
                * self.PI
                * 1j
                / self.LAMBDA
                * (z + (x ** 2) / 2 / z)
            )
        )
        return fresnel_propagator * np.fft.fftshift(np.fft.fft(output_wave * fresnel_phase))

    def Tabg(self, F):
        """Функция пропускания"""
        return np.sqrt(self.DELTA * F * self.attenuation_length() / (self.DELTA * F * self.attenuation_length() + self.absorption_aperture(F)**2))

if __name__ == "__main__":
    root = tk.Tk()
    app = XRayLensApp(root)
    root.mainloop()
