
#  Deep Learning Visualizer

> 一个极致直观、零依赖的深度学习与 CNN 交互式可视化实验室。
> 打破干巴巴的数学公式，让张量流动、卷积核滑动、特征浓缩和过拟合过程“肉眼可见”。

##  在线体验 (Live Demo)

**[点击这里进入在线实验室](https://cecelia6712.github.io/deep-learning-visualizer/)** 
*(建议在 PC 端浏览器全屏体验，配合右侧导航栏食用更佳)*

##  核心特性 (Features)

- **学术极简风**：纯白背景、衬线体与等宽体搭配，低饱和度高亮，沉浸式推导体验。
- **零依赖单文件**：无需安装 Node.js 或任何框架，原生 HTML + CSS + JS 编写，双击即用。
- **手写步进器**：拒绝简陋的下拉框，所有参数（Kernel、Stride、Padding）皆可精准微调。
- **全景视图联动**：绝不展示“没头没尾”的局部切片，所有计算动画都带有原图与全局上下文。

##  已实现模块 (第 4 章 - 卷积神经网络)

目前已完成《动手学深度学习》第 4 章的核心概念可视化，包含 6 个独立实验模块：

1. **01 二维互相关**：逐项相乘、柱状图累加、滑动窗口全景高亮。
2. **02 多通道与 1×1 卷积**：解析 Cin/Cout 独立关系、叠层卡片、通道交叉投票网络。
3. **03 池化层**：Max/Avg 实时切换、重叠（卷积）与不重叠（池化）对比。
4. **04 Padding 与 Stride**：公式动态代入、补零虚线与滑动轨迹实时联动。
5. **05 LeNet 张量流水线**：2.5D 特征图卡片、真实的点阵特征、扁平化拉长与降维全景。
6. **06 过拟合实验台**：Canvas 动态曲线、过拟合裂口展示、早停/Dropout/数据增强解药。

##  技术栈 (Tech Stack)

- **HTML5 / CSS3**：重度使用 CSS Grid 绘制独立可交互的矩阵格子；原生 Flex 布局。
- **Vanilla JavaScript (ES5+)**：手写 SPA 路由框架 (`Lab.register`)，完全隔离各模块状态。
- **Canvas 2D**：仅用于过拟合实验台的动态折线图绘制，配合 `ResizeObserver` 完美适配。

##  如何本地运行

1. Clone 本仓库到本地：
   ```bash
   git clone https://github.com/Cecelia6712/deep-learning-visualizer.git
