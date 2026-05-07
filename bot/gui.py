"""
Dark-themed trading dashboard GUI built with tkinter.
Mirrors the layout from the screenshot: header stats bar, ticker strip,
tabbed main panel (Positions / History / Logs / Perf), right sidebar with
strategy params + ML engine status, and bottom LONG/SHORT action buttons.
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, font as tkfont
from typing import Callable, Dict, List, Optional

# ── Colour palette ────────────────────────────────────────────────────────────
BG         = "#080d16"
BG_PANEL   = "#0d1525"
BG_CARD    = "#111c2e"
BG_HEADER  = "#0a1020"
BORDER     = "#1c2d45"
GREEN      = "#00e676"
GREEN_DIM  = "#00c853"
RED        = "#ff1744"
RED_DIM    = "#d50000"
BLUE       = "#2979ff"
BLUE_DIM   = "#1565c0"
GOLD       = "#ffd600"
CYAN       = "#00e5ff"
PURPLE     = "#ea80fc"
TEXT       = "#e8eaf6"
TEXT_DIM   = "#546e8a"
TEXT_MID   = "#8fa8c8"
ML_COLOR   = "#64b5f6"   # blue label for ML-set params
ACCENT     = "#00e676"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _colored(val: float, positive_color: str = GREEN, negative_color: str = RED) -> str:
    return positive_color if val >= 0 else negative_color


# ── Main Application ──────────────────────────────────────────────────────────

class BotGUI:
    """
    Launch with BotGUI(engine).run().
    The engine is expected to expose:
        engine.start() / engine.stop()
        engine.risk_manager.open_positions  (dict)
        engine.trainer.models               (dict)
        engine._trade_log                   (list)
        engine.print_stats()
    Callbacks are injected into the engine via engine.gui_callback.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._log_lines: List[str] = []

        # Tk root MUST exist before any tk.Variable is created
        self.root = tk.Tk()

        self._selected_pair = tk.StringVar()
        self._param_vars: Dict[str, tk.Variable] = {}

        self._setup_root()
        self._build_ui()
        self._apply_theme()

    # ── Root window ──────────────────────────────────────────────────────────

    def _setup_root(self) -> None:
        self.root.title("CONFLUENCE BOT — Coinbase Spot")
        self.root.configure(bg=BG)
        self.root.geometry("1400x820")
        self.root.minsize(1100, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Custom fonts
        self._font_mono  = tkfont.Font(family="Consolas",   size=9)
        self._font_small = tkfont.Font(family="Consolas",   size=8)
        self._font_big   = tkfont.Font(family="Consolas",   size=14, weight="bold")
        self._font_title = tkfont.Font(family="Consolas",   size=11, weight="bold")
        self._font_label = tkfont.Font(family="Consolas",   size=9)
        self._font_stat  = tkfont.Font(family="Consolas",   size=18, weight="bold")

    # ── Full UI assembly ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_top_bar()
        self._build_stats_bar()
        self._build_ticker_bar()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_main_panel(body)
        self._build_side_panel(body)

        self._build_bottom_bar()

    # ── Top bar ──────────────────────────────────────────────────────────────

    def _build_top_bar(self) -> None:
        bar = tk.Frame(self.root, bg=BG_HEADER, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Lightning icon + title
        icon_lbl = tk.Label(bar, text="⚡", bg=BG_HEADER, fg=GREEN,
                            font=tkfont.Font(family="Consolas", size=14, weight="bold"))
        icon_lbl.pack(side="left", padx=(10, 0))

        title = tk.Label(bar, text="CONFLUENCE BOT", bg=BG_HEADER, fg=TEXT,
                         font=self._font_title)
        title.pack(side="left", padx=(4, 16))

        # Status badge
        self._status_badge = _Badge(bar, text="SIMULATION", color=BG_HEADER, fg=GOLD)
        self._status_badge.pack(side="left", padx=4)

        # Running indicator
        self._running_badge = _Badge(bar, text="● INITIALIZING", color=BG_HEADER, fg=TEXT_DIM)
        self._running_badge.pack(side="left", padx=4)

        # Strategy description
        self._strategy_lbl = tk.Label(
            bar, text="LSTM + XGBoost + LightGBM + RandomForest  ·  Coinbase Spot",
            bg=BG_HEADER, fg=TEXT_DIM, font=self._font_small
        )
        self._strategy_lbl.pack(side="left", padx=12)

        # STOP button (right)
        self._stop_btn = tk.Button(
            bar, text="  STOP  ", bg=RED, fg=TEXT,
            font=self._font_title, relief="flat", cursor="hand2",
            activebackground=RED_DIM, activeforeground=TEXT,
            command=self._on_stop,
        )
        self._stop_btn.pack(side="right", padx=10, pady=6)

        # START button
        self._start_btn = tk.Button(
            bar, text="  START  ", bg=GREEN_DIM, fg=BG,
            font=self._font_title, relief="flat", cursor="hand2",
            activebackground=GREEN, activeforeground=BG,
            command=self._on_start,
        )
        self._start_btn.pack(side="right", padx=(0, 4), pady=6)

    # ── Stats bar ────────────────────────────────────────────────────────────

    def _build_stats_bar(self) -> None:
        bar = tk.Frame(self.root, bg=BG_PANEL, height=72)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        stats = [
            ("BALANCE",   "$10,000.00", TEXT,    "_lbl_balance"),
            ("TOTAL P&L", "$0.00",      GREEN,   "_lbl_pnl"),
            ("WIN RATE",  "0.0%",       TEXT,    "_lbl_winrate"),
            ("TRADES",    "0",          TEXT,    "_lbl_trades"),
            ("FEES PAID", "$0.00",      TEXT,    "_lbl_fees"),
            ("30D VOL",   "$0",         TEXT,    "_lbl_vol"),
            ("SHARPE",    "—",          TEXT,    "_lbl_sharpe"),
        ]

        for label, val, color, attr in stats:
            cell = tk.Frame(bar, bg=BG_PANEL, padx=18)
            cell.pack(side="left", fill="y", pady=8)

            tk.Label(cell, text=label, bg=BG_PANEL, fg=TEXT_DIM,
                     font=self._font_small).pack(anchor="w")
            lbl = tk.Label(cell, text=val, bg=BG_PANEL, fg=color,
                           font=self._font_stat)
            lbl.pack(anchor="w")
            setattr(self, attr, lbl)

            # Separator
            sep = tk.Frame(bar, bg=BORDER, width=1)
            sep.pack(side="left", fill="y", pady=4)

    # ── Ticker bar ───────────────────────────────────────────────────────────

    def _build_ticker_bar(self) -> None:
        bar = tk.Frame(self.root, bg=BG_CARD, height=32)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        self._ticker_frame = tk.Frame(bar, bg=BG_CARD)
        self._ticker_frame.pack(side="left", fill="y", padx=4)

        self._ticker_labels: Dict[str, tk.Label] = {}
        self._pair_btns: Dict[str, tk.Button] = {}

    def _refresh_ticker(self, prices: Dict[str, float]) -> None:
        # Remove old labels
        for w in self._ticker_frame.winfo_children():
            w.destroy()
        self._ticker_labels.clear()
        self._pair_btns.clear()

        sel = self._selected_pair.get()
        for pair, price in prices.items():
            short = pair.replace("-USD", "")
            is_sel = (pair == sel)
            bg = BG if is_sel else BG_CARD
            fg = GREEN if is_sel else TEXT_MID

            btn = tk.Button(
                self._ticker_frame,
                text=f"{short} ${price:,.0f}" if price > 100 else f"{short} ${price:.4f}",
                bg=bg, fg=fg, relief="flat", cursor="hand2",
                font=self._font_small, activebackground=BG_PANEL,
                command=lambda p=pair: self._select_pair(p),
                padx=6,
            )
            btn.pack(side="left", fill="y")
            self._pair_btns[pair] = btn
            self._ticker_labels[pair] = btn

    def _select_pair(self, pair: str) -> None:
        self._selected_pair.set(pair)
        self._refresh_ticker_selection()
        self._update_bottom_buttons()

    def _refresh_ticker_selection(self) -> None:
        sel = self._selected_pair.get()
        for pair, btn in self._pair_btns.items():
            if pair == sel:
                btn.configure(bg=BG, fg=GREEN)
            else:
                btn.configure(bg=BG_CARD, fg=TEXT_MID)

    # ── Main panel (left) ────────────────────────────────────────────────────

    def _build_main_panel(self, parent: tk.Frame) -> None:
        self._main_frame = tk.Frame(parent, bg=BG)
        self._main_frame.pack(side="left", fill="both", expand=True)

        nb_frame = tk.Frame(self._main_frame, bg=BG_PANEL)
        nb_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Custom tab bar
        tab_bar = tk.Frame(nb_frame, bg=BG_PANEL, height=32)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)

        self._tab_content = tk.Frame(nb_frame, bg=BG)
        self._tab_content.pack(fill="both", expand=True)

        self._tab_frames: Dict[str, tk.Frame] = {}
        self._tab_btns: Dict[str, tk.Button] = {}

        tabs = ["CHART", "POSITIONS", "HISTORY", "LOGS", "PERF"]
        for tab in tabs:
            btn = tk.Button(
                tab_bar, text=tab, bg=BG_PANEL, fg=TEXT_DIM,
                relief="flat", cursor="hand2", font=self._font_label,
                activebackground=BG, activeforeground=GREEN,
                padx=14, pady=6,
                command=lambda t=tab: self._switch_tab(t),
            )
            btn.pack(side="left")
            self._tab_btns[tab] = btn

            frame = tk.Frame(self._tab_content, bg=BG)
            self._tab_frames[tab] = frame

        self._build_chart_tab()
        self._build_positions_tab()
        self._build_history_tab()
        self._build_logs_tab()
        self._build_perf_tab()

        self._switch_tab("POSITIONS")

    def _switch_tab(self, name: str) -> None:
        for n, f in self._tab_frames.items():
            f.pack_forget()
        for n, b in self._tab_btns.items():
            b.configure(fg=TEXT_DIM, bg=BG_PANEL)
        self._tab_frames[name].pack(fill="both", expand=True)
        self._tab_btns[name].configure(fg=GREEN, bg=BG)

    # CHART tab — placeholder (future: embedded matplotlib candlestick)
    def _build_chart_tab(self) -> None:
        f = self._tab_frames["CHART"]
        tk.Label(
            f, text="Chart coming soon\n(select a pair from the ticker bar)",
            bg=BG, fg=TEXT_DIM, font=self._font_mono
        ).pack(expand=True)

    # POSITIONS tab
    def _build_positions_tab(self) -> None:
        f = self._tab_frames["POSITIONS"]
        cols = ("Pair", "Side", "Entry", "SL", "TP", "Size $", "Conf", "Unr. P&L")
        self._pos_tree = _make_tree(f, cols)
        self._pos_tree.pack(fill="both", expand=True)
        self._pos_empty = tk.Label(f, text="No open positions", bg=BG, fg=TEXT_DIM,
                                   font=self._font_mono)

    # HISTORY tab
    def _build_history_tab(self) -> None:
        f = self._tab_frames["HISTORY"]
        cols = ("Time", "Pair", "Side", "Entry", "Exit", "P&L %", "Reason", "Size $")
        self._hist_tree = _make_tree(f, cols)
        self._hist_tree.pack(fill="both", expand=True)

    # LOGS tab
    def _build_logs_tab(self) -> None:
        f = self._tab_frames["LOGS"]
        scroll = tk.Scrollbar(f, bg=BG_PANEL, troughcolor=BG)
        self._log_text = tk.Text(
            f, bg=BG, fg=TEXT_MID, font=self._font_small,
            wrap="word", state="disabled", yscrollcommand=scroll.set,
            insertbackground=TEXT, selectbackground=BLUE_DIM,
        )
        scroll.configure(command=self._log_text.yview)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

        # Tag colours for log levels
        self._log_text.tag_configure("INFO",    foreground=TEXT_MID)
        self._log_text.tag_configure("WARNING", foreground=GOLD)
        self._log_text.tag_configure("ERROR",   foreground=RED)
        self._log_text.tag_configure("ENTRY",   foreground=GREEN)
        self._log_text.tag_configure("EXIT",    foreground=CYAN)

    # PERF tab
    def _build_perf_tab(self) -> None:
        f = self._tab_frames["PERF"]
        self._perf_vars: Dict[str, tk.StringVar] = {}
        metrics = [
            ("Total Trades",     "_perf_trades"),
            ("Win Rate",         "_perf_winrate"),
            ("Avg Win %",        "_perf_avg_win"),
            ("Avg Loss %",       "_perf_avg_loss"),
            ("Profit Factor",    "_perf_pf"),
            ("Max Drawdown",     "_perf_mdd"),
            ("Sharpe Ratio",     "_perf_sharpe"),
            ("Expectancy $/t",   "_perf_exp"),
        ]
        grid = tk.Frame(f, bg=BG)
        grid.pack(padx=20, pady=20, anchor="nw")
        for row, (label, attr) in enumerate(metrics):
            tk.Label(grid, text=label, bg=BG, fg=TEXT_DIM,
                     font=self._font_label, width=18, anchor="w").grid(
                row=row, column=0, padx=(0, 12), pady=3, sticky="w")
            var = tk.StringVar(value="—")
            lbl = tk.Label(grid, textvariable=var, bg=BG, fg=TEXT,
                           font=self._font_mono, anchor="w")
            lbl.grid(row=row, column=1, pady=3, sticky="w")
            self._perf_vars[attr] = var

    # ── Side panel (right) ──────────────────────────────────────────────────

    def _build_side_panel(self, parent: tk.Frame) -> None:
        side = tk.Frame(parent, bg=BG_PANEL, width=300)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        canvas = tk.Canvas(side, bg=BG_PANEL, highlightthickness=0)
        vsb = tk.Scrollbar(side, orient="vertical", command=canvas.yview, bg=BG_PANEL)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG_PANEL)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        self._build_strategy_params(inner)
        _sep(inner)
        self._build_ml_engine_section(inner)
        _sep(inner)
        self._build_trade_learning(inner)
        _sep(inner)
        self._build_all_pairs(inner)

    def _build_strategy_params(self, parent: tk.Frame) -> None:
        _section_header(parent, "STRATEGY PARAMS")

        params = [
            # (label,            attr_key,           default, min, max,  is_ml, is_float)
            ("ATR Multiplier",  "atr_mult",          1.5,    0.5, 4.0,   True,  True),
            ("SL ATR Mult",     "sl_atr_mult",       1.5,    0.5, 3.0,   True,  True),
            ("TP ATR Mult",     "tp_atr_mult",       2.5,    1.0, 6.0,   True,  True),
            ("BB Period",       "bb_period",         20,     5,   50,    False, False),
            ("BB Std Dev",      "bb_std",            2,      1,   4,     False, False),
            ("RSI Period",      "rsi_period",        14,     5,   30,    False, False),
            ("RSI Overbought",  "rsi_ob",            70,     60,  90,    True,  False),
            ("RSI Oversold",    "rsi_os",            30,     10,  40,    True,  False),
            ("Position Size %", "pos_size",          2,      0.5, 10,    True,  True),
            ("Max Positions",   "max_pos",           3,      1,   10,    False, False),
            ("Min Confidence",  "min_conf",          62,     50,  95,    True,  False),
            ("Retrain (hrs)",   "retrain_hrs",       24,     1,   168,   False, False),
        ]

        for label, key, default, lo, hi, is_ml, is_float in params:
            row = tk.Frame(parent, bg=BG_PANEL)
            row.pack(fill="x", padx=10, pady=1)

            ml_tag = tk.Label(row, text="ML", bg=BG_PANEL, fg=ML_COLOR,
                              font=self._font_small) if is_ml else tk.Label(
                row, text="   ", bg=BG_PANEL, font=self._font_small)
            ml_tag.pack(side="right", padx=(0, 2))

            var = tk.DoubleVar(value=default) if is_float else tk.IntVar(value=int(default))
            val_lbl = tk.Label(row, textvariable=var, bg=BG_PANEL, fg=ML_COLOR if is_ml else TEXT,
                               font=self._font_small, width=6, anchor="e")
            val_lbl.pack(side="right")

            tk.Label(row, text=label, bg=BG_PANEL, fg=TEXT_MID,
                     font=self._font_small, anchor="w").pack(side="left")

            slider = tk.Scale(
                parent, from_=lo, to=hi, resolution=0.01 if is_float else 1,
                orient="horizontal", variable=var,
                bg=BG_PANEL, fg=TEXT_DIM, troughcolor=BORDER,
                highlightthickness=0, showvalue=False, sliderlength=12,
                command=lambda v, k=key: self._on_param_change(k, v),
            )
            slider.pack(fill="x", padx=10, pady=0)
            self._param_vars[key] = var

        # VWAP checkbox
        vwap_row = tk.Frame(parent, bg=BG_PANEL)
        vwap_row.pack(fill="x", padx=10, pady=4)
        self._vwap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            vwap_row, text="VWAP Filter Enabled", variable=self._vwap_var,
            bg=BG_PANEL, fg=TEXT_MID, selectcolor=BG, activebackground=BG_PANEL,
            font=self._font_small,
        ).pack(side="left")

    def _on_param_change(self, key: str, value: str) -> None:
        """Push param changes into live engine config."""
        from bot.config import config
        mapping = {
            "sl_atr_mult": "sl_atr_multiplier",
            "tp_atr_mult": "tp_atr_multiplier",
            "max_pos":     "max_open_positions",
            "min_conf":    None,   # handled separately (percentage)
            "retrain_hrs": "retrain_interval_hours",
        }
        if key == "min_conf":
            config.min_confidence = float(value) / 100
        elif key in mapping and mapping[key]:
            setattr(config, mapping[key], float(value) if "." in str(value) else int(float(value)))

    def _build_ml_engine_section(self, parent: tk.Frame) -> None:
        _section_header(parent, "ML ENGINE")

        self._ml_status_lbl = tk.Label(parent, text="Initializing...", bg=BG_PANEL,
                                       fg=TEXT_DIM, font=self._font_small)
        self._ml_status_lbl.pack(anchor="w", padx=10)

        checks = [
            ("ML Signal Boost",   "_ml_boost_var",    True),
            ("Auto-Adjust Params","_ml_auto_var",      True),
        ]
        for label, attr, default in checks:
            var = tk.BooleanVar(value=default)
            setattr(self, attr, var)
            tk.Checkbutton(
                parent, text=label, variable=var,
                bg=BG_PANEL, fg=TEXT_MID, selectcolor=BG,
                activebackground=BG_PANEL, font=self._font_small,
            ).pack(anchor="w", padx=10)

        self._ml_detail_lbl = tk.Label(parent, text="", bg=BG_PANEL, fg=TEXT_DIM,
                                       font=self._font_small, justify="left")
        self._ml_detail_lbl.pack(anchor="w", padx=10, pady=2)

    def _build_trade_learning(self, parent: tk.Frame) -> None:
        _section_header(parent, "TRADE LEARNING")
        self._learn_lbl = tk.Label(
            parent,
            text="Memory: 0 trades (0W / 0L)\nBlocked: 0 bad setups\nNeed 10 more trades to adapt",
            bg=BG_PANEL, fg=TEXT_DIM, font=self._font_small, justify="left",
        )
        self._learn_lbl.pack(anchor="w", padx=10, pady=4)

    def _build_all_pairs(self, parent: tk.Frame) -> None:
        _section_header(parent, f"ALL PAIRS ({0})")
        self._pairs_container = tk.Frame(parent, bg=BG_PANEL)
        self._pairs_container.pack(fill="x", padx=6, pady=4)
        self._pair_rows: Dict[str, tk.Frame] = {}

    def _refresh_all_pairs(self, prices: Dict[str, float]) -> None:
        for w in self._pairs_container.winfo_children():
            w.destroy()
        for pair, price in prices.items():
            row = tk.Frame(self._pairs_container, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            short = pair.replace("-USD", "")
            tk.Label(row, text=short, bg=BG_PANEL, fg=TEXT_MID,
                     font=self._font_small, width=6, anchor="w").pack(side="left")
            price_str = f"${price:>10,.2f}" if price > 1 else f"${price:.6f}"
            fg = GREEN if self.engine.risk_manager.has_open_position(pair) else TEXT_DIM
            tk.Label(row, text=price_str, bg=BG_PANEL, fg=fg,
                     font=self._font_small, anchor="e").pack(side="right")

    # ── Bottom bar ───────────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> None:
        bar = tk.Frame(self.root, bg=BG, height=52)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._long_btn = tk.Button(
            bar, text="↑  LONG —", bg=GREEN_DIM, fg=BG,
            font=self._font_title, relief="flat", cursor="hand2",
            activebackground=GREEN, activeforeground=BG,
            command=self._manual_long,
        )
        self._long_btn.pack(side="left", fill="both", expand=True)

        self._short_btn = tk.Button(
            bar, text="↓  SHORT —", bg=RED_DIM, fg=TEXT,
            font=self._font_title, relief="flat", cursor="hand2",
            activebackground=RED, activeforeground=TEXT,
            command=self._manual_short,
        )
        self._short_btn.pack(side="right", fill="both", expand=True)

        self._update_bottom_buttons()

    def _update_bottom_buttons(self) -> None:
        pair = self._selected_pair.get()
        short = pair.replace("-USD", "") if pair else "—"
        self._long_btn.configure(text=f"↑  LONG {short}")
        self._short_btn.configure(text=f"↓  SHORT {short}")

    def _manual_long(self) -> None:
        self.append_log(f"[MANUAL] LONG {self._selected_pair.get()} requested", "ENTRY")

    def _manual_short(self) -> None:
        self.append_log(f"[MANUAL] SHORT {self._selected_pair.get()} requested", "EXIT")

    # ── Theme ────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=BG_CARD, foreground=TEXT, fieldbackground=BG_CARD,
                        rowheight=22, font=("Consolas", 8))
        style.configure("Treeview.Heading",
                        background=BG_PANEL, foreground=TEXT_DIM,
                        font=("Consolas", 8, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", BLUE_DIM)])
        style.configure("Vertical.TScrollbar", background=BG_PANEL, troughcolor=BG)

    # ── Public update API (called from background thread) ────────────────────

    def post(self, fn: Callable, *args) -> None:
        """Thread-safe way to schedule a GUI update."""
        self._queue.put((fn, args))

    def update_stats(
        self,
        balance: float,
        pnl: float,
        win_rate: float,
        trades: int,
        fees: float,
        vol: float = 0,
        sharpe: float = 0,
    ) -> None:
        self.post(self._do_update_stats, balance, pnl, win_rate, trades, fees, vol, sharpe)

    def _do_update_stats(self, balance, pnl, win_rate, trades, fees, vol, sharpe) -> None:
        self._lbl_balance.configure(text=f"${balance:,.2f}")
        pnl_color = GREEN if pnl >= 0 else RED
        self._lbl_pnl.configure(text=f"${pnl:+,.2f}", fg=pnl_color)
        wr_color = GREEN if win_rate >= 60 else (GOLD if win_rate >= 50 else RED)
        self._lbl_winrate.configure(text=f"{win_rate:.1f}%", fg=wr_color)
        self._lbl_trades.configure(text=str(trades))
        self._lbl_fees.configure(text=f"${fees:,.2f}")
        self._lbl_vol.configure(text=f"${vol:,.0f}")
        self._lbl_sharpe.configure(text=f"{sharpe:.2f}" if sharpe else "—")

    def update_prices(self, prices: Dict[str, float]) -> None:
        self.post(self._do_update_prices, prices)

    def _do_update_prices(self, prices: Dict[str, float]) -> None:
        if not self._selected_pair.get() and prices:
            self._selected_pair.set(next(iter(prices)))
        self._refresh_ticker(prices)
        self._refresh_all_pairs(prices)
        self._update_bottom_buttons()

    def update_positions(self) -> None:
        self.post(self._do_update_positions)

    def _do_update_positions(self) -> None:
        tree = self._pos_tree
        for row in tree.get_children():
            tree.delete(row)

        positions = self.engine.risk_manager.open_positions
        if not positions:
            self._pos_empty.pack(expand=True)
        else:
            self._pos_empty.pack_forget()

        # Use cached prices — no network calls from the GUI thread
        cached = getattr(self.engine, "_last_prices", {})
        for pid, p in positions.items():
            cur = cached.get(pid, p.entry_price)
            unr = (cur - p.entry_price) / p.entry_price * 100 * (1 if p.side == "BUY" else -1)
            tag = "pos" if unr >= 0 else "neg"
            tree.insert("", "end", values=(
                pid, p.side,
                f"{p.entry_price:.4f}", f"{p.stop_loss:.4f}", f"{p.take_profit:.4f}",
                f"${p.position_size_quote:.2f}", f"{p.confidence:.2%}",
                f"{unr:+.2f}%",
            ), tags=(tag,))

        tree.tag_configure("pos", foreground=GREEN)
        tree.tag_configure("neg", foreground=RED)

    def update_history(self) -> None:
        self.post(self._do_update_history)

    def _do_update_history(self) -> None:
        tree = self._hist_tree
        for row in tree.get_children():
            tree.delete(row)

        exits = [t for t in self.engine._trade_log if t["action"] == "EXIT"]
        for t in reversed(exits[-200:]):
            pnl = t.get("pnl_pct", 0) or 0
            tag = "pos" if pnl >= 0 else "neg"
            ts = t.get("timestamp", "")[:19]
            tree.insert("", "end", values=(
                ts, t["product_id"], t["side"],
                f"{t.get('price', 0):.4f}", f"{t.get('price', 0):.4f}",
                f"{pnl:+.2f}%", t.get("order_id", "")[:10],
                f"${t.get('size_usd', 0):.2f}",
            ), tags=(tag,))
        tree.tag_configure("pos", foreground=GREEN)
        tree.tag_configure("neg", foreground=RED)

    def update_ml_status(self, status: str, detail: str = "") -> None:
        self.post(self._do_update_ml_status, status, detail)

    def _do_update_ml_status(self, status: str, detail: str) -> None:
        self._ml_status_lbl.configure(text=status, fg=GREEN if "Active" in status else GOLD)
        self._ml_detail_lbl.configure(text=detail)

    def update_running_status(self, running: bool, paper: bool) -> None:
        self.post(self._do_update_running_status, running, paper)

    def _do_update_running_status(self, running: bool, paper: bool) -> None:
        if running:
            label = "● RUNNING (SIMULATION)" if paper else "● RUNNING (LIVE)"
            color = GOLD if paper else GREEN
        else:
            label = "● STOPPED"
            color = RED
        self._running_badge.configure(text=label, fg=color)
        badge_text = "SIMULATION" if paper else "LIVE"
        self._status_badge.configure(text=badge_text, fg=GOLD if paper else RED)

    def update_perf(self, metrics: Dict) -> None:
        self.post(self._do_update_perf, metrics)

    def _do_update_perf(self, metrics: Dict) -> None:
        for key, var in self._perf_vars.items():
            if key in metrics:
                var.set(str(metrics[key]))

    def update_learn_status(self, wins: int, losses: int, blocked: int) -> None:
        self.post(self._do_update_learn, wins, losses, blocked)

    def _do_update_learn(self, wins: int, losses: int, blocked: int) -> None:
        total = wins + losses
        need = max(0, 10 - total)
        self._learn_lbl.configure(
            text=f"Memory: {total} trades ({wins}W / {losses}L)\n"
                 f"Blocked: {blocked} bad setups\n"
                 f"{'Adapting from trade history' if total >= 10 else f'Need {need} more trades to learn'}"
        )

    def append_log(self, text: str, level: str = "INFO") -> None:
        self.post(self._do_append_log, text, level)

    def _do_append_log(self, text: str, level: str) -> None:
        widget = self._log_text
        widget.configure(state="normal")
        tag = level if level in ("INFO", "WARNING", "ERROR", "ENTRY", "EXIT") else "INFO"
        widget.insert("end", text + "\n", tag)
        widget.see("end")
        widget.configure(state="disabled")

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._running:
            return
        self.append_log("Starting bot...", "INFO")
        self._start_btn.configure(state="disabled")
        threading.Thread(target=self._run_engine, daemon=True).start()

    def _on_stop(self) -> None:
        self.append_log("Stopping bot...", "WARNING")
        self._running = False
        try:
            self.engine.stop()
        except Exception:
            pass
        self.update_running_status(False, True)
        self._start_btn.configure(state="normal")

    def _run_engine(self) -> None:
        self._running = True
        from bot.config import config
        self.update_running_status(True, config.paper_trading)
        try:
            self.engine.start()
        except Exception as e:
            self.append_log(f"Engine error: {e}", "ERROR")
        finally:
            self._running = False
            self.update_running_status(False, False)
            self._start_btn.configure(state="normal")

    def _on_close(self) -> None:
        self._on_stop()
        self.root.destroy()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _process_queue(self) -> None:
        try:
            while True:
                fn, args = self._queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

    def run(self) -> None:
        self.root.after(100, self._process_queue)
        self.root.mainloop()


# ── Widget helpers ─────────────────────────────────────────────────────────────

class _Badge(tk.Label):
    def __init__(self, parent, text: str, color: str, fg: str, **kw):
        super().__init__(parent, text=f" {text} ", bg=color, fg=fg,
                         font=tkfont.Font(family="Consolas", size=8, weight="bold"),
                         padx=4, pady=2, relief="flat", **kw)


def _section_header(parent: tk.Frame, title: str) -> None:
    row = tk.Frame(parent, bg=BG_PANEL)
    row.pack(fill="x", padx=6, pady=(10, 2))
    tk.Label(row, text=title, bg=BG_PANEL, fg=TEXT_MID,
             font=tkfont.Font(family="Consolas", size=8, weight="bold")).pack(side="left")
    tk.Frame(row, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=6)


def _sep(parent: tk.Frame) -> None:
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)


def _make_tree(parent: tk.Frame, columns: tuple) -> ttk.Treeview:
    frame = tk.Frame(parent, bg=BG)
    frame.pack(fill="both", expand=True)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    vsb.pack(side="right", fill="y")

    tree = ttk.Treeview(frame, columns=columns, show="headings",
                        yscrollcommand=vsb.set, selectmode="browse")
    vsb.configure(command=tree.yview)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=90, anchor="center")

    tree.pack(fill="both", expand=True)
    return tree
