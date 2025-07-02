import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
import json
import os

DATA_FILE = 'data.json'
README_FILE = 'README.md'
FIELDS = [
    'year', 'title', 'venue', 'abstract', 'citation', 'code', 'paper_url', 'project', 'demo', 'supplementary'
]

class PaperManagerTk:
    def __init__(self, root):
        self.root = root
        self.root.title('论文管理器 (tkinter)')
        self.papers = []
        self.filtered = []
        self.create_widgets()
        self.load_data()
        self.refresh_table()

    def create_widgets(self):
        """
        创建主界面，包括按钮区、搜索区和表格区，所有区域均自适应布局。
        按钮区用于操作（添加、编辑、删除、保存、生成README、重新加载），
        搜索区用于关键词检索，表格区动态根据json字段生成。
        """
        self.root.configure(bg='#f5f6fa')
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Treeview',
                        background='#f5f6fa',
                        foreground='#222',
                        rowheight=28,
                        fieldbackground='#f5f6fa',
                        font=('微软雅黑', 11))
        style.configure('Treeview.Heading', font=('微软雅黑', 12, 'bold'), background='#dcdde1', foreground='#273c75')
        style.map('Treeview', background=[('selected', '#d6e4ff')])

        frm = tk.Frame(self.root, bg='#f5f6fa')
        frm.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        # 按钮区
        btnfrm = tk.Frame(frm, bg='#f5f6fa')
        btnfrm.grid(row=0, column=0, sticky='ew', pady=6)
        frm.grid_columnconfigure(0, weight=1)
        btn_style = {'font': ('微软雅黑', 11), 'bg': '#40739e', 'fg': 'white', 'activebackground': '#273c75', 'activeforeground': 'white', 'relief': tk.GROOVE, 'bd': 1, 'padx': 8, 'pady': 2}
        tk.Button(btnfrm, text='添加', command=self.add_paper, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btnfrm, text='编辑', command=self.edit_paper, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btnfrm, text='删除', command=self.delete_paper, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btnfrm, text='保存', command=self.save_data, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btnfrm, text='生成README', command=self.generate_readme, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btnfrm, text='重新加载', command=self.load_data, **btn_style).pack(side=tk.LEFT, padx=2)
        btnfrm.grid_columnconfigure(tuple(range(6)), weight=1)
        # 搜索区
        searchfrm = tk.Frame(frm, bg='#f5f6fa')
        searchfrm.grid(row=1, column=0, sticky='ew', pady=6)
        frm.grid_rowconfigure(1, weight=0)
        self.search_var = tk.StringVar()
        tk.Label(searchfrm, text='🔍', font=('微软雅黑', 12), bg='#f5f6fa').pack(side=tk.LEFT, padx=(0,2))
        search_entry = tk.Entry(searchfrm, textvariable=self.search_var, font=('微软雅黑', 11), width=40, relief=tk.GROOVE, bd=2)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        searchfrm.grid_columnconfigure(1, weight=1)
        tk.Button(searchfrm, text='搜索', command=self.search_papers, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(searchfrm, text='重置', command=self.reset_search, **btn_style).pack(side=tk.LEFT, padx=2)
        # 表格区
        treefrm = tk.Frame(frm, bg='#f5f6fa')
        treefrm.grid(row=2, column=0, sticky='nsew', pady=(8,0))
        frm.grid_rowconfigure(2, weight=1)
        # 动态根据json数据的字段生成表头和列宽
        self.treefrm = treefrm
        self._init_tree(self.treefrm)
    def _init_tree(self, parent):
        """
        动态初始化表格区，根据json数据实际字段生成表头和列宽。
        支持新增字段自动出现在表格中。
        parent: 表格父容器Frame。
        """
        # 动态获取所有字段
        all_fields = set(FIELDS)
        # 尝试从data.json读取所有可能的字段
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                for p in papers:
                    all_fields.update(p.keys())
        except Exception:
            pass
        self.dynamic_fields = list(FIELDS)
        for f in all_fields:
            if f not in self.dynamic_fields:
                self.dynamic_fields.append(f)
        # 创建Treeview表格
        self.tree = ttk.Treeview(parent, columns=self.dynamic_fields, show='headings', height=16, style='Treeview')
        for f in self.dynamic_fields:
            width = 120 if f in ['title','author','venue'] else 90
            self.tree.heading(f, text=f)
            self.tree.column(f, width=width, anchor='w')
        # 垂直滚动条
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        # 双击表格行可编辑
        self.tree.bind('<Double-1>', lambda e: self.edit_paper())

    def load_data(self):
        """
        从data.json加载论文数据，失败时弹窗警告。
        加载后刷新表格。
        """
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                self.papers = json.load(f)
        except Exception as e:
            self.papers = []
            messagebox.showwarning('警告', f'加载 data.json 失败: {e}')
        self.filtered = self.papers.copy()
        self.refresh_table()

    def save_data(self):
        """
        保存当前论文数据到data.json，保存成功或失败均弹窗提示。
        保存时按照年份降序、同一年内按论文标题首字母升序排序。
        """
        try:
            # 先排序：年份降序，标题首字母升序
            def sort_key(p):
                # 年份优先，降序；标题首字母升序
                year = p.get('year', '')
                try:
                    year_int = int(year)
                except Exception:
                    year_int = 0
                title = p.get('title', '')
                return (-year_int, title.lower())
            sorted_papers = sorted(self.papers, key=sort_key)
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(sorted_papers, f, ensure_ascii=False, indent=2)
            messagebox.showinfo('提示', '保存成功!')
        except Exception as e:
            messagebox.showwarning('警告', f'保存失败: {e}')

    def refresh_table(self):
        """
        刷新表格内容，若有新字段则重建表格。
        """
        self.tree.delete(*self.tree.get_children())
        # 若有新字段，重建表格
        all_fields = set(FIELDS)
        for p in self.filtered:
            all_fields.update(p.keys())
        if set(self.dynamic_fields) != all_fields:
            # 重新生成表格
            self.tree.destroy()
            self._init_tree(self.treefrm)
        for idx, paper in enumerate(self.filtered):
            values = [paper.get(f, '') for f in self.dynamic_fields]
            self.tree.insert('', 'end', iid=idx, values=values)

    def get_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def add_paper(self):
        data = self.edit_dialog()
        if data:
            self.papers.append(data)
            self.filtered = self.papers.copy()
            self.refresh_table()

    def edit_paper(self):
        idx = self.get_selected_index()
        if idx is None or idx >= len(self.filtered):
            messagebox.showwarning('警告', '请先选择要编辑的论文')
            return
        paper = self.filtered[idx]
        data = self.edit_dialog(paper)
        if data:
            # 找到原始 papers 中的索引
            orig_idx = self.papers.index(paper)
            self.papers[orig_idx] = data
            self.filtered[idx] = data
            self.refresh_table()

    def delete_paper(self):
        idx = self.get_selected_index()
        if idx is None or idx >= len(self.filtered):
            messagebox.showwarning('警告', '请先选择要删除的论文')
            return
        if messagebox.askyesno('确认', '确定要删除该论文吗?'):
            paper = self.filtered[idx]
            self.papers.remove(paper)
            self.filtered.remove(paper)
            self.refresh_table()

    def search_papers(self):
        keyword = self.search_var.get().strip().lower()
        if not keyword:
            self.filtered = self.papers.copy()
        else:
            self.filtered = []
            for paper in self.papers:
                for f in FIELDS:
                    if keyword in str(paper.get(f, '')).lower():
                        self.filtered.append(paper)
                        break
        self.refresh_table()

    def reset_search(self):
        self.search_var.set('')
        self.filtered = self.papers.copy()
        self.refresh_table()

    def generate_readme(self):
        """
        按照项目README.md的格式生成README.md文件。
        """
        try:
            year_map = {}
            for p in self.papers:
                y = str(p.get('year', 'earlier'))
                year_map.setdefault(y, []).append(p)
            # 年份排序，2025、2024、2023、2022、earlier
            def year_sort_key(y):
                try:
                    return -int(y)
                except Exception:
                    return 9999
            years = sorted(year_map.keys(), key=year_sort_key)
            lines = [
                '# Awesome Monocular Depth Estimation',
                '',
                'A curated list of monocular depth estimation papers.',
                '',
                'The list focuses primarily on papers published after 2022, including some particularly outstanding work from earlier years.',
                '',
                '精选单目深度估计论文列表。精选并整理了 `2022` 年后发表的单目深度估计论文，同时涵盖部分早期的优秀成果。',
                ''
            ]
            for y in years:
                lines.append(f'## {y}\n')
                for p in year_map[y]:
                    # 标题
                    title = p.get('title', '')
                    # 会议/期刊
                    venue = p.get('venue', '')
                    # 论文链接
                    paper_url = p.get('paper_url', '')
                    # 代码链接
                    code = p.get('code', '')
                    # 项目页
                    project = p.get('project', '')
                    # demo
                    demo = p.get('demo', '')
                    # 补充材料
                    supp = p.get('supplementary', '')
                    # 摘要
                    abstract = p.get('abstract', '')
                    # bibtex
                    citation = p.get('citation', '')
                    # 构建标题行
                    title_line = f"### [{title}]({paper_url})" if paper_url else f"### {title}"
                    if venue:
                        title_line += f" ![Static Badge](https://img.shields.io/badge/{venue}-FF0000)"
                    lines.append(title_line)
                    # 资源链接
                    link_line = []
                    if code:
                        link_line.append(f"[Code]({code})")
                    if project:
                        link_line.append(f"[Project]({project})")
                    if demo:
                        link_line.append(f"[Demo]({demo})")
                    if supp:
                        link_line.append(f"[Supplementary]({supp})")
                    if link_line:
                        lines.append(' | '.join(link_line) + ' ')
                    # 摘要
                    if abstract:
                        lines.append('<details closed>')
                        lines.append('<summary>Abstract</summary>')
                        lines.append(abstract.strip())
                        lines.append('</details>')
                    # bibtex
                    if citation:
                        lines.append('')
                        lines.append('<details closed>')
                        lines.append('<summary>Citation</summary>')
                        lines.append('')
                        lines.append('```bibtex')
                        lines.append(citation.strip())
                        lines.append('```')
                        lines.append('</details>')
                    lines.append('')
            with open(README_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            messagebox.showinfo('提示', 'README.md 生成成功!')
        except Exception as e:
            messagebox.showwarning('警告', f'生成 README.md 失败: {e}')

    def edit_dialog(self, paper=None):
        # 动态字段
        all_fields = set(FIELDS)
        if paper:
            all_fields.update(paper.keys())
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                for p in papers:
                    all_fields.update(p.keys())
        except Exception:
            pass
        fields = list(FIELDS)
        for f in all_fields:
            if f not in fields:
                fields.append(f)
        dlg = tk.Toplevel(self.root)
        dlg.title('编辑论文' if paper else '添加论文')
        dlg.configure(bg='#f5f6fa')
        frm = tk.Frame(dlg, bg='#f5f6fa')
        frm.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        dlg.grid_rowconfigure(0, weight=1)
        dlg.grid_columnconfigure(0, weight=1)
        entries = {}
        for i, f in enumerate(fields):
            tk.Label(frm, text=f, bg='#f5f6fa', font=('微软雅黑', 10)).grid(row=i, column=0, sticky='w', pady=2)
            if f in ['abstract', 'citation']:
                ent = tk.Text(frm, width=40, height=3, font=('微软雅黑', 10))
                if paper and f in paper:
                    ent.insert('1.0', paper[f])
            else:
                ent = tk.Entry(frm, width=45, font=('微软雅黑', 10))
                if paper and f in paper:
                    ent.insert(0, paper[f])
            ent.grid(row=i, column=1, padx=2, pady=2, sticky='ew')
            frm.grid_rowconfigure(i, weight=1)
            frm.grid_columnconfigure(1, weight=1)
            entries[f] = ent
        result = {}
        def on_ok():
            for f, ent in entries.items():
                if isinstance(ent, tk.Text):
                    val = ent.get('1.0', 'end').strip()
                else:
                    val = ent.get().strip()
                if val:
                    result[f] = val
            dlg.destroy()
        btnfrm = tk.Frame(frm, bg='#f5f6fa')
        btnfrm.grid(row=len(fields), column=0, columnspan=2, pady=(8,0), sticky='ew')
        btn_style = {'font': ('微软雅黑', 10), 'bg': '#40739e', 'fg': 'white', 'activebackground': '#273c75', 'activeforeground': 'white', 'relief': tk.GROOVE, 'bd': 1, 'padx': 8, 'pady': 2}
        tk.Button(btnfrm, text='确定', command=on_ok, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(btnfrm, text='取消', command=dlg.destroy, **btn_style).pack(side=tk.LEFT, padx=4)
        frm.grid_rowconfigure(len(fields), weight=0)
        frm.grid_columnconfigure(1, weight=1)
        dlg.grab_set()
        dlg.wait_window()
        return result if result else None

def main():
    root = tk.Tk()
    app = PaperManagerTk(root)
    root.mainloop()

if __name__ == '__main__':
    main()
