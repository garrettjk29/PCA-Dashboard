import queue
import threading
import tkinter as tk
from tkinter import messagebox

import matplotlib
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure

from getDataFunction import get_daily_prices
from pcaAnalysis import build_returns_df, run_pca
from storage import load_portfolio, save_portfolio

DEFAULT_WINDOW_DAYS = "365"
DATE_FORMAT = "%Y-%m-%d"


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dashboard")
        self.geometry("1150x700")

        self.results_data = {}  # ticker -> prices
        self._active_fetches = 0
        self._queue = queue.Queue()
        self._date_anchor = "end"  # which of start/end drives the window calc

        self._build_status_bar()
        self._build_main_area()
        self._build_search_row()
        self._build_results_area()
        self._build_analysis_area()

        self._load_saved_portfolio()

        self.after(100, self._poll_queue)

    def _load_saved_portfolio(self):
        prices = load_portfolio()
        for ticker, ticker_prices in prices.items():
            self._add_result_row(ticker, ticker_prices, save=False)

    STATUS_IDLE_COLOR = "#2ecc71"
    STATUS_BUSY_COLOR = "#e74c3c"

    COLOR_SURFACE = "#fcfcfb"
    COLOR_PRIMARY_INK = "#0b0b0b"
    COLOR_MUTED_INK = "#898781"
    COLOR_GRIDLINE = "#e1e0d9"
    COLOR_SEQUENTIAL = "#2a78d6"
    COLOR_DIVERGING_NEG = "#e34948"
    COLOR_DIVERGING_MID = "#f0efec"
    COLOR_DIVERGING_POS = "#2a78d6"

    def _build_status_bar(self):
        self.status_bar = tk.Frame(
            self, height=5, bg=self.STATUS_IDLE_COLOR, highlightthickness=0, bd=0
        )
        self.status_bar.pack(fill="x", side="top")
        self.status_bar.pack_propagate(False)

    def _build_main_area(self):
        main = tk.Frame(self)
        main.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(main)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(main)
        self.right_frame.pack(side="left", fill="both", expand=True)

    def _build_search_row(self):
        row = tk.Frame(self.left_frame)
        row.pack(fill="x", padx=15, pady=(15, 10))

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(row, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, ipady=4)
        search_entry.bind("<Return>", self._on_search_enter)

    def _build_results_area(self):
        container = tk.Frame(self.left_frame, relief="solid", borderwidth=1)
        container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.results_frame = tk.Frame(container)
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_window_controls(self):
        row = tk.Frame(self.right_frame)
        row.pack(fill="x", padx=(0, 15), pady=(15, 0))

        tk.Label(row, text="Window (days):").pack(side="left")
        self.window_days_var = tk.StringVar(value=DEFAULT_WINDOW_DAYS)
        window_entry = tk.Entry(row, textvariable=self.window_days_var, width=6)
        window_entry.pack(side="left", padx=(4, 12))
        window_entry.bind("<Return>", self._on_window_days_enter)

        tk.Label(row, text="Start:").pack(side="left")
        self.start_date_var = tk.StringVar()
        start_entry = tk.Entry(row, textvariable=self.start_date_var, width=10)
        start_entry.pack(side="left", padx=(4, 12))
        start_entry.bind("<Return>", self._on_start_date_enter)

        tk.Label(row, text="End:").pack(side="left")
        self.end_date_var = tk.StringVar()
        end_entry = tk.Entry(row, textvariable=self.end_date_var, width=10)
        end_entry.pack(side="left", padx=(4, 0))
        end_entry.bind("<Return>", self._on_end_date_enter)

    def _on_window_days_enter(self, event=None):
        self._recalculate_window_from_anchor()
        self._update_analysis()

    def _on_start_date_enter(self, event=None):
        self._date_anchor = "start"
        self._recalculate_window_from_anchor()
        self._update_analysis()

    def _on_end_date_enter(self, event=None):
        self._date_anchor = "end"
        self._recalculate_window_from_anchor()
        self._update_analysis()

    def _recalculate_window_from_anchor(self):
        """Given N days and one of start/end, fill in the other."""
        try:
            days = int(self.window_days_var.get())
        except ValueError:
            return

        if self._date_anchor == "start":
            start_str = self.start_date_var.get().strip()
            if not start_str:
                return
            try:
                start = pd.Timestamp(start_str)
            except ValueError:
                return
            end = start + pd.Timedelta(days=days)
        else:
            end_str = self.end_date_var.get().strip()
            if end_str:
                try:
                    end = pd.Timestamp(end_str)
                except ValueError:
                    return
            elif self.results_data:
                end = build_returns_df(self.results_data).index.max()
            else:
                return
            start = end - pd.Timedelta(days=days)

        self.start_date_var.set(start.strftime(DATE_FORMAT))
        self.end_date_var.set(end.strftime(DATE_FORMAT))

    def _build_analysis_area(self):
        self._build_window_controls()

        matrix_container = tk.LabelFrame(self.right_frame, text="PCA Loadings")
        matrix_container.pack(fill="both", expand=True, padx=(0, 15), pady=(10, 10))

        self.matrix_frame = tk.Frame(matrix_container)
        self.matrix_frame.pack(fill="both", expand=True, padx=5, pady=5)

        chart_container = tk.LabelFrame(self.right_frame, text="Explained Variance")
        chart_container.pack(fill="both", expand=True, padx=(0, 15), pady=(0, 15))

        self.chart_frame = tk.Frame(chart_container)
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _update_analysis(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        if len(self.results_data) < 2:
            message = "Add at least 2 tickers to run PCA."
            tk.Label(self.matrix_frame, text=message, fg=self.COLOR_MUTED_INK).pack(
                expand=True
            )
            tk.Label(self.chart_frame, text=message, fg=self.COLOR_MUTED_INK).pack(
                expand=True
            )
            return

        try:
            if not self.start_date_var.get().strip() or not self.end_date_var.get().strip():
                self._recalculate_window_from_anchor()

            returns_df = build_returns_df(self.results_data)
            windowed = returns_df.loc[self.start_date_var.get():self.end_date_var.get()]
            results = run_pca(windowed)
        except Exception as exc:
            message = f"PCA failed: {exc}"
            tk.Label(
                self.matrix_frame, text=message, fg=self.COLOR_MUTED_INK, wraplength=280
            ).pack(expand=True)
            tk.Label(
                self.chart_frame, text=message, fg=self.COLOR_MUTED_INK, wraplength=280
            ).pack(expand=True)
            return

        self._draw_loading_matrix(results)
        self._draw_variance_chart(results)

    def _draw_loading_matrix(self, results):
        tickers = results["tickers"]
        loadings = results["loading_matrix"]
        n_tickers, n_pcs = loadings.shape
        pc_labels = [f"PC{i + 1}" for i in range(n_pcs)]

        fig = Figure(figsize=(4, 3), dpi=100, facecolor=self.COLOR_SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COLOR_SURFACE)

        cmap = LinearSegmentedColormap.from_list(
            "diverging",
            [self.COLOR_DIVERGING_NEG, self.COLOR_DIVERGING_MID, self.COLOR_DIVERGING_POS],
        )
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        ax.imshow(loadings, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(n_pcs))
        ax.set_xticklabels(pc_labels, color=self.COLOR_MUTED_INK, fontsize=8)
        ax.set_yticks(range(n_tickers))
        ax.set_yticklabels(tickers, color=self.COLOR_MUTED_INK, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(n_tickers):
            for j in range(n_pcs):
                value = loadings[i, j]
                text_color = self.COLOR_PRIMARY_INK if abs(value) < 0.6 else "#ffffff"
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color,
                )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_variance_chart(self, results):
        evr = results["evr"]
        pc_labels = [f"PC{i + 1}" for i in range(len(evr))]

        fig = Figure(figsize=(4, 3), dpi=100, facecolor=self.COLOR_SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.COLOR_SURFACE)

        bars = ax.bar(pc_labels, evr * 100, color=self.COLOR_SEQUENTIAL, width=0.6)
        ax.set_ylabel("Variance explained (%)", color=self.COLOR_MUTED_INK, fontsize=8)
        ax.tick_params(colors=self.COLOR_MUTED_INK, labelsize=8, length=0)
        ax.yaxis.grid(True, color=self.COLOR_GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(self.COLOR_GRIDLINE)

        for bar, value in zip(bars, evr):
            ax.annotate(
                f"{value * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=self.COLOR_PRIMARY_INK,
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_search_enter(self, event=None):
        ticker = self.search_var.get().strip().upper()
        if not ticker:
            return
        self.search_var.set("")

        self._active_fetches += 1
        self.status_bar.config(bg=self.STATUS_BUSY_COLOR)

        threading.Thread(
            target=self._fetch_worker, args=(ticker,), daemon=True
        ).start()

    def _fetch_worker(self, ticker):
        try:
            prices = get_daily_prices(ticker, "all")
            self._queue.put(("success", ticker, prices))
        except Exception as exc:
            self._queue.put(("error", ticker, str(exc)))

    def _poll_queue(self):
        try:
            while True:
                status, ticker, payload = self._queue.get_nowait()
                self._active_fetches = max(0, self._active_fetches - 1)
                if self._active_fetches == 0:
                    self.status_bar.config(bg=self.STATUS_IDLE_COLOR)

                if status == "success":
                    self._add_result_row(ticker.upper(), payload)
                else:
                    messagebox.showerror("Data fetch failed", f"{ticker}: {payload}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _add_result_row(self, ticker, prices, save=True):
        self.results_data[ticker] = prices
        if save:
            save_portfolio(self.results_data)

        row = tk.Frame(self.results_frame)
        row.pack(fill="x", pady=2)

        label = tk.Label(row, text=ticker, anchor="w")
        label.pack(side="left", fill="x", expand=True)

        def on_delete():
            self.results_data.pop(ticker, None)
            save_portfolio(self.results_data)
            row.destroy()
            self._update_analysis()

        delete_btn = tk.Button(row, text="Delete", command=on_delete)
        delete_btn.pack(side="right")

        self._update_analysis()


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
