import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser

from send_email import send_email, _load_env_file


QQ_HELP_URL = "https://service.mail.qq.com/"


class EmailGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("简单邮件发送器 (Gmail)")
        self.geometry("720x640")
        self.attachments = []

        _load_env_file()

        # Defaults (prefilled for Gmail)
        default_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        default_port = int(os.environ.get("SMTP_PORT", "465"))
        default_from = os.environ.get("FROM_EMAIL", os.environ.get("SMTP_USERNAME", ""))
        default_user = os.environ.get("SMTP_USERNAME", default_from)

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Helper: labeled entry
        def add_labeled(parent, label, default="", show=None):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(frame, textvariable=var, show=show)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return var, entry

        # From/Username/Password
        self.from_var, _ = add_labeled(container, "发件邮箱", default_from)
        self.user_var, _ = add_labeled(container, "SMTP 用户名", default_user)
        self.pass_var, _ = add_labeled(container, "应用专用密码", show="*")

        # To, Subject
        self.to_var, _ = add_labeled(container, "收件人", "")
        self.subj_var, _ = add_labeled(container, "主题", "")

        # SMTP config
        cfg = ttk.Frame(container)
        cfg.pack(fill=tk.X, pady=6)
        ttk.Label(cfg, text="SMTP 服务器", width=14).pack(side=tk.LEFT)
        self.server_var = tk.StringVar(value=default_server)
        ttk.Entry(cfg, textvariable=self.server_var, width=20).pack(side=tk.LEFT)

        ttk.Label(cfg, text="端口").pack(side=tk.LEFT, padx=(8, 2))
        self.port_var = tk.IntVar(value=default_port)
        ttk.Entry(cfg, textvariable=self.port_var, width=6).pack(side=tk.LEFT)

        self.starttls_var = tk.BooleanVar(value=(default_port == 587))
        ttk.Checkbutton(cfg, text="使用 STARTTLS (587)", variable=self.starttls_var, command=self._toggle_tls_defaults).pack(side=tk.LEFT, padx=10)

        # Attachments
        attach_frame = ttk.Frame(container)
        attach_frame.pack(fill=tk.X, pady=6)
        ttk.Label(attach_frame, text="附件", width=14).pack(side=tk.LEFT)
        self.attach_label = ttk.Label(attach_frame, text="无")
        self.attach_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(attach_frame, text="添加附件", command=self._add_attachments).pack(side=tk.LEFT, padx=6)
        ttk.Button(attach_frame, text="清空附件", command=self._clear_attachments).pack(side=tk.LEFT)

        # Body
        ttk.Label(container, text="正文").pack(anchor=tk.W, pady=(8, 2))
        self.body_txt = tk.Text(container, height=16)
        self.body_txt.pack(fill=tk.BOTH, expand=True)

        # Actions
        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="Gmail 应用专用密码指引", command=self._open_help).pack(side=tk.LEFT)
        self.send_btn = ttk.Button(btns, text="发送", command=self._send_async)
        self.send_btn.pack(side=tk.RIGHT)

        # Info banner
        info = (
            "使用 Gmail：需先在 Google 账号中开启两步验证，并创建‘应用专用密码’；"
            "将 16 位应用专用密码填入上方‘应用专用密码’（不是你的 Google 登录密码）。"
        )
        ttk.Label(container, text=info, foreground="#444").pack(fill=tk.X, pady=(8, 0))

    def _toggle_tls_defaults(self):
        if self.starttls_var.get():
            if self.port_var.get() == 465:
                self.port_var.set(587)
        else:
            if self.port_var.get() == 587:
                self.port_var.set(465)

    def _open_help(self):
        try:
            webbrowser.open("https://support.google.com/accounts/answer/185833?hl=zh-Hans")
        except Exception:
            messagebox.showinfo("提示", "请在浏览器中搜索：Gmail 应用专用密码 创建 方法")

    def _add_attachments(self):
        paths = filedialog.askopenfilenames(title="选择附件")
        if paths:
            self.attachments.extend(paths)
            self._refresh_attach_label()

    def _clear_attachments(self):
        self.attachments = []
        self._refresh_attach_label()

    def _refresh_attach_label(self):
        if not self.attachments:
            self.attach_label.config(text="无")
        else:
            names = [os.path.basename(p) for p in self.attachments]
            self.attach_label.config(text=", ".join(names))

    def _send_async(self):
        t = threading.Thread(target=self._send, daemon=True)
        self.send_btn.config(state=tk.DISABLED, text="发送中...")
        t.start()

    def _send(self):
        try:
            from_email = self.from_var.get().strip()
            username = self.user_var.get().strip() or from_email
            password = self.pass_var.get()
            to_raw = self.to_var.get().strip()
            to_emails = [e.strip() for e in to_raw.replace(";", ",").split(",") if e.strip()]
            subject = self.subj_var.get().strip()
            body = self.body_txt.get("1.0", tk.END)
            server = self.server_var.get().strip()
            port = int(self.port_var.get())
            use_starttls = bool(self.starttls_var.get())

            if not (from_email and username and password and to_emails and subject):
                raise ValueError("请填写发件邮箱、用户名、授权码、收件人和主题")

            send_email(
                smtp_server=server,
                smtp_port=port,
                username=username,
                password=password,
                from_email=from_email,
                to_emails=to_emails,
                subject=subject,
                body=body,
                attachments=self.attachments,
                use_starttls=use_starttls,
            )

            self.after(0, lambda: messagebox.showinfo("成功", "邮件已发送"))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda m=err_msg: messagebox.showerror("发送失败", m))
        finally:
            self.after(0, lambda: self.send_btn.config(state=tk.NORMAL, text="发送"))


if __name__ == "__main__":
    app = EmailGUI()
    app.mainloop()
