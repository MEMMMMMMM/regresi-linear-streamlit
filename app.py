import pandas as pd
import numpy as np
import ttkbootstrap as tb
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog, Listbox, MULTIPLE, Canvas, Toplevel, Frame, Label, Scrollbar
from sklearn.linear_model import LinearRegression
from scipy.stats import t as tdist, f as fdist


df = None


# =====================================================================
# FUNGSI TRANSFORMASI LOG
# =====================================================================
def ln_transform(series):
    return np.log(series + 1)


# =====================================================================
# PILIH FILE CSV
# =====================================================================
def choose_file():
    global df
    filepath = filedialog.askopenfilename(
        title="Pilih file CSV", filetypes=[("CSV Files", "*.csv")]
    )
    if not filepath:
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    columns = list(df.columns)

    y_combo.configure(values=columns)
    y_combo.set("")

    x_listbox.delete(0, "end")
    for col in columns:
        x_listbox.insert("end", col)

    Messagebox.show_info("Dataset berhasil dimuat!", "Sukses")


# =====================================================================
# RUN REGRESSION
# =====================================================================
def run_regression():
    global df
    if df is None:
        Messagebox.show_error("Upload CSV terlebih dahulu!", "Error")
        return

    # Ambil input user
    y_var = y_combo.get()
    x_indices = x_listbox.curselection()

    if not y_var:
        Messagebox.show_error("Pilih variabel Y!", "Error")
        return
    if len(x_indices) == 0:
        Messagebox.show_error("Pilih minimal 1 variabel X!", "Error")
        return

    # Alpha
    try:
        alpha = float(alpha_entry.get())
    except:
        Messagebox.show_error("Nilai alpha tidak valid!", "Error")
        return

    X_vars = [x_listbox.get(i) for i in x_indices]
    data = df[[y_var] + X_vars].dropna()

    X = data[X_vars].astype(float)
    y = data[y_var].astype(float)

    # Transformasi jika dipilih
    used_transform = []

    if chk_log_y.instate(["selected"]):
        y = ln_transform(y)
        used_transform.append("log(Y)")

    if chk_log_x.instate(["selected"]):
        X = X.apply(ln_transform)
        used_transform.append("log(X)")

    transform_label = ", ".join(used_transform) if used_transform else "Tidak digunakan"

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    n = len(X)
    k = len(X_vars)

    # Variance & F-test
    sse = np.sum(residuals ** 2)
    ssr = np.sum((y_pred - np.mean(y)) ** 2)
    mse = sse / (n - k - 1)

    Xmat = np.column_stack([np.ones(n), X])
    cov = mse * np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(cov))

    intercept_se = se[0]
    coef_se = se[1:]

    intercept_t = model.intercept_ / intercept_se
    intercept_p = 2 * (1 - tdist.cdf(np.abs(intercept_t), df=n - k - 1))

    t_values = model.coef_ / coef_se
    p_values = 2 * (1 - tdist.cdf(np.abs(t_values), df=n - k - 1))

    r2 = model.score(X, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    msr = ssr / k
    f_stat = msr / mse
    f_pval = 1 - fdist.cdf(f_stat, k, n - k - 1)

    # =====================================================================
    # WINDOW OUTPUT DENGAN TAB
    # =====================================================================
    win = Toplevel()
    win.title("Hasil Regresi")
    win.geometry("1200x750")
    win.configure(bg="#1E3A55")

    notebook = tb.Notebook(win, bootstyle="dark")
    notebook.pack(fill="both", expand=True, pady=10)

    tab_summary = Frame(notebook, bg="#1E3A55")
    tab_detail = Frame(notebook, bg="#1E3A55")

    notebook.add(tab_summary, text="Summary")
    notebook.add(tab_detail, text="Detail Variabel")

    # =====================================================================
    # TAB SUMMARY
    # =====================================================================
    Label(tab_summary, text="Hasil Regresi Linier Berganda",
          font=("Segoe UI", 22, "bold"), fg="white", bg="#1E3A55").pack(pady=10)

    Label(tab_summary, text=f"Transformasi: {transform_label}",
          font=("Segoe UI", 12), fg="white", bg="#1E3A55").pack()

    Label(tab_summary, text=f"Y: {y_var}", font=("Segoe UI", 12),
          fg="white", bg="#1E3A55").pack()

    Label(tab_summary, text=f"X: {', '.join(X_vars)}",
          font=("Segoe UI", 12), fg="white", bg="#1E3A55").pack(pady=(0, 20))

    Label(tab_summary, text=f"Intercept = {model.intercept_:.4f}",
          font=("Segoe UI", 14, "bold"), fg="white", bg="#1E3A55").pack(pady=10)

    sm = Frame(tab_summary, bg="#1E3A55")
    sm.pack(anchor="w", padx=40, pady=20)

    Label(sm, text="Model Summary", font=("Segoe UI", 16, "bold"),
          fg="white", bg="#1E3A55").pack(anchor="w")

    Label(sm, text=f"R² = {r2:.4f}", fg="white", bg="#1E3A55").pack(anchor="w")
    Label(sm, text=f"Adjusted R² = {adj_r2:.4f}", fg="white", bg="#1E3A55").pack(anchor="w")
    Label(sm, text=f"F-statistic = {f_stat:.4f}", fg="white", bg="#1E3A55").pack(anchor="w")
    Label(sm, text=f"F p-value = {f_pval:.4f}", fg="white", bg="#1E3A55").pack(anchor="w")
    Label(sm, text=f"n = {n}", fg="white", bg="#1E3A55").pack(anchor="w")

    # =====================================================================
    # TAB DETAIL VARIABEL
    # =====================================================================
    grid_frame = Frame(tab_detail, bg="#1E3A55")
    grid_frame.pack(pady=20)

    max_cols = 3
    row, col = 0, 0

    for var, beta, tval, pval in zip(X_vars, model.coef_, t_values, p_values):

        card = Frame(grid_frame, bg="#2C2F33", padx=20, pady=10)
        card.grid(row=row, column=col, padx=25, pady=20)

        Label(card, text=var, font=("Segoe UI", 12, "bold"),
              fg="white", bg="#2C2F33").pack(anchor="w")

        Label(card, text=f"β = {beta:.4f}", fg="white", bg="#2C2F33").pack(anchor="w")
        Label(card, text=f"t = {tval:.4f}", fg="white", bg="#2C2F33").pack(anchor="w")

        emoji = "(Good)" if pval < alpha else "(Bad)"
        Label(card, text=f"p = {pval:.4f}   {emoji}",
              fg="white", bg="#2C2F33").pack(anchor="w")

        col += 1
        if col >= max_cols:
            col = 0
            row += 1


# =====================================================================
# UI APPLICATION
# =====================================================================
app = tb.Window(themename="superhero")
app.title("Analisis Regresi Linear Berganda")
app.geometry("950x900")
app.resizable(False, False)

tb.Label(app, text="Analisis Regresi Linier Berganda",
         font=("Segoe UI", 30, "bold"), bootstyle="light").pack(pady=25)

canvas_frame = tb.Frame(app)
canvas_frame.pack()

canvas = Canvas(canvas_frame, width=880, height=760,
                bg=app.cget("background"), highlightthickness=0)
canvas.pack()


def round_rect(x1, y1, x2, y2, r, color):
    canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90,
                      fill=color, outline=color)
    canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90,
                      fill=color, outline=color)
    canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90,
                      fill=color, outline=color)
    canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90,
                      fill=color, outline=color)
    canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline=color)
    canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline=color)


round_rect(10, 10, 860, 740, r=50, color="#2C2F33")

style = tb.Style()
style.configure("Black.TFrame", background="#2C2F33")

content = tb.Frame(canvas_frame, style="Black.TFrame")
content.place(relx=0.5, y=380, anchor="center")

# UPLOAD CSV
btn_upload = tb.Button(content, text="Upload CSV", width=20,
                       bootstyle="info", command=choose_file)
btn_upload.grid(row=0, column=0, pady=(10, 25))

# Y
tb.Label(content, text="Pilih Variabel Y:",
         font=("Segoe UI", 12, "bold"), bootstyle="light").grid(row=1, column=0, sticky="w")
y_combo = tb.Combobox(content, width=70)
y_combo.grid(row=2, column=0, pady=7)

# X
tb.Label(content, text="Pilih Variabel X (bisa lebih dari satu):",
         font=("Segoe UI", 12, "bold"), bootstyle="light").grid(row=3, column=0, sticky="w", pady=(20, 7))

x_listbox = Listbox(content, width=70, height=12, selectmode=MULTIPLE)
x_listbox.grid(row=4, column=0, pady=5)

# Alpha + Transformasi
transform_frame = tb.Frame(content, style="Black.TFrame")
transform_frame.grid(row=5, column=0, pady=20, sticky="w")

tb.Label(transform_frame, text="Alpha:",
         font=("Segoe UI", 12, "bold"), bootstyle="light").grid(row=0, column=0, padx=5)

alpha_entry = tb.Entry(transform_frame, width=10)
alpha_entry.insert(0, "0.05")
alpha_entry.grid(row=0, column=1, padx=5)

chk_log_y = tb.Checkbutton(transform_frame, text="Log(Y)", bootstyle="info")
chk_log_y.grid(row=0, column=2, padx=10)

chk_log_x = tb.Checkbutton(transform_frame, text="Log(X)", bootstyle="info")
chk_log_x.grid(row=0, column=3, padx=10)

# RUN BUTTON
btn_run = tb.Button(content, text="Jalankan Regresi",
                    width=30, bootstyle="success",
                    command=run_regression)
btn_run.grid(row=6, column=0, pady=25)

app.mainloop()
