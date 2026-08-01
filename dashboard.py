import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from getDataFunction import _TIMEFRAME_MAP, get_daily_prices
from storage import load_portfolio, save_portfolio

TIMEFRAMES = list(_TIMEFRAME_MAP.keys())


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dashboard")
        self.geometry("1150x700")

        self.results_data = {}  # ticker -> prices
        self.timeframes = {}  # ticker -> timeframe it was fetched with
        self._active_fetches = 0
        self._queue = queue.Queue()

        self._build_status_bar()
        self._build_search_row()
        self._build_results_area()

        self._load_saved_portfolio()

        self.after(100, self._poll_queue)

    def _load_saved_portfolio(self):
        prices, timeframes = load_portfolio()
        self.timeframes = timeframes
        for ticker, ticker_prices in prices.items():
            self._add_result_row(ticker, ticker_prices, save=False)

    STATUS_IDLE_COLOR = "#2ecc71"
    STATUS_BUSY_COLOR = "#e74c3c"

    def _build_status_bar(self):
        self.status_bar = tk.Frame(
            self, height=5, bg=self.STATUS_IDLE_COLOR, highlightthickness=0, bd=0
        )
        self.status_bar.pack(fill="x", side="top")
        self.status_bar.pack_propagate(False)

    def _build_search_row(self):
        row = tk.Frame(self)
        row.pack(fill="x", padx=15, pady=(15, 10))

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(row, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, ipady=4)
        search_entry.bind("<Return>", self._on_search_enter)

        self.timeframe_var = tk.StringVar(value=TIMEFRAMES[0])
        timeframe_dropdown = ttk.Combobox(
            row,
            textvariable=self.timeframe_var,
            values=TIMEFRAMES,
            state="readonly",
            width=10,
        )
        timeframe_dropdown.pack(side="left", padx=(10, 0))

    def _build_results_area(self):
        container = tk.Frame(self, relief="solid", borderwidth=1)
        container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.results_frame = tk.Frame(container)
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _on_search_enter(self, event=None):
        ticker = self.search_var.get().strip().upper()
        if not ticker:
            return
        timeframe = self.timeframe_var.get()
        self.search_var.set("")

        self._active_fetches += 1
        self.status_bar.config(bg=self.STATUS_BUSY_COLOR)

        threading.Thread(
            target=self._fetch_worker, args=(ticker, timeframe), daemon=True
        ).start()

    def _fetch_worker(self, ticker, timeframe):
        try:
            prices = get_daily_prices(ticker, timeframe)
            self._queue.put(("success", ticker, timeframe, prices))
        except Exception as exc:
            self._queue.put(("error", ticker, timeframe, str(exc)))

    def _poll_queue(self):
        try:
            while True:
                status, ticker, timeframe, payload = self._queue.get_nowait()
                self._active_fetches = max(0, self._active_fetches - 1)
                if self._active_fetches == 0:
                    self.status_bar.config(bg=self.STATUS_IDLE_COLOR)

                if status == "success":
                    self._add_result_row(ticker.upper(), payload, timeframe)
                else:
                    messagebox.showerror("Data fetch failed", f"{ticker}: {payload}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _add_result_row(self, ticker, prices, timeframe=None, save=True):
        self.results_data[ticker] = prices
        if timeframe is not None:
            self.timeframes[ticker] = timeframe
        if save:
            save_portfolio(self.results_data, self.timeframes)

        row = tk.Frame(self.results_frame)
        row.pack(fill="x", pady=2)

        label = tk.Label(row, text=ticker, anchor="w")
        label.pack(side="left", fill="x", expand=True)

        def on_delete():
            self.results_data.pop(ticker, None)
            self.timeframes.pop(ticker, None)
            save_portfolio(self.results_data, self.timeframes)
            row.destroy()

        delete_btn = tk.Button(row, text="Delete", command=on_delete)
        delete_btn.pack(side="right")


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
