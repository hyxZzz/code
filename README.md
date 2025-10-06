# code

## 将本地修改推送到 GitHub 的步骤
1. 确认工作区干净并查看当前分支：
   ```bash
   git status -sb
   ```
2. 若尚未连接远程仓库，请添加你的 GitHub 仓库地址（将 `<your-repo-url>` 替换为实际地址）：
   ```bash
   git remote add origin <your-repo-url>
   ```
   如果已经存在 `origin`，可使用 `git remote set-url origin <your-repo-url>` 更新。
3. 将所有变更提交到当前分支：
   ```bash
   git add -A
   git commit -m "说明本次修改内容"
   ```
4. 如果想要推送到 GitHub 的主分支（通常为 `main` 或 `master`），先确保本地分支名称一致。可以通过以下命令重命名当前分支：
   ```bash
   git branch -M main
   ```
   或者在推送时指定远程分支名称，例如：
   ```bash
   git push -u origin work
   ```
5. 将提交推送到远程仓库：
   ```bash
   git push -u origin main
   ```
   如果分支不同，请将 `main` 替换为目标分支名。
6. 推送成功后，就可以在 GitHub 仓库页面上看到最新的代码变更。

> **提示**：首次推送时可能需要输入 GitHub 账号密码或 Personal Access Token。建议提前在 GitHub 账户设置中创建 Token，并在命令行中使用。
