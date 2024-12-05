# 备查表2：如何高效获取 AI 相关的信息

### 备查表2：如何高效获取 AI 相关的信息

目前 AI 相关的信息非常多，但是99%的信息都不值得你去跟踪，都是二手三手信息，你需要做的是找到信息源头，用有限打败无限。

#### 通用

##### Github：快速获取开源项目

网址：https://github.com。

GitHub是全球最大的代码托管和协作平台，拥有大量GPT和LLM相关的开源项目和活跃的开发者社区，可以帮助开发者快速获取最新技术、参与项目并交流学习。

使用方法如下：

**技巧1：通过关键词和限定forks或者stars数量，从中筛选项目**

搜索created:>2023-03-01 stars:>=1000 label:gpt，即2023年3月1日后发布，关注量超过1000，关键词为GPT的项目，超过2000个。这2000多个项目，几乎代表了LLM领域最优质的开源项目，也代表了未来技术趋势。

但是这些项目数量太多，我们无法一一查看，那么我们可以通过下面方法缩小跟踪范围。

以下项目截止时间是2024年8月12日。

| 搜索关键词                                                   | 项目数量（个） | 链接                                                         |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| created:>2023-03-01 stars:>=1000 label:gpt                   | 2000+          | https://github.com/search?q=created%3A%3E2023-03-01+stars%3A%3E%3D1000+label%3Agpt&type=Repositories&ref=advsearch&l=&l=&s=stars&o=desc |
| llm forks:>500                                               | 163            | https://github.com/search?q=llm+forks%3A%3E500+&type=repositories |
| gpt forks:>500                                               | 152            | https://github.com/search?q=gpt+forks%3A%3E500+&type=repositories |
| created:"> 2023-03-15" stars:>500 forks:>500 label:"gpt OR LLM OR rag OR Finetuning OR Gen AI " | 489            | https://github.com/search?q=created%3A%22%3E+2023-03-15%22+stars%3A%3E500+forks%3A%3E500+label%3A%22gpt+OR+LLM+OR+rag+OR+Finetuning+OR+GenAI%22&type=Repositories&ref=advsearch&l=&l= |

**项目推荐：**

Ollama：本地运行大模型管家，使用Ollama可以实现在本地运行大模型。支持主流大模型，还可以上传自己的大模型在Ollama上。项目地址：https://github.com/Ollama/Ollama

**技巧2：标签搜索法**

除了搜索标签外，还可以点击开源项目的标签，发现标签下的项目。进入后，可以选择多种方式进行排序。以下试举一些典型的：

- "#llm"。地址：https://github.com/topics/llm?o=desc&s=stars
- "#AI"。地址：https://github.com/topics/AI
- "#rag"。地址：https://github.com/topics/rag
- "#fine-tuning"。地址：https://github.com/topics/fine-tuning
- "#agents"。地址：https://github.com/topics/agents

例如，根据标星数量和更新时间，可以筛选出关注度最高的RAG项目。你可以近似认为，这些RAG项目，代表了市场上最主流的RAG项目。

| 序号 | 名称               | 简介                                                         | GitHub 仓库地址                                      |
| ---- | ------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| 1    | Dify               | 开源LLM应用开发平台，提供AI工作流、RAG管道等直观界面。       | https://github.com/langgenius/dify                   |
| 2    | Quivr              | 你的第二大脑，利用 GenerativeAI 的力量成为您的个人助理！     | https://github.com/QuivrHQ/quivr                     |
| 3    | Open WebUI         | 面向 LLM 的用户友好型 WebUI                                  | https://github.com/open-webui/open-webui             |
| 4    | Langchain-Chatchat | 基于Langchain和ChatGLM等语言模型的RAG和Agent应用。           | https://github.com/chatchat-space/Langchain-Chatchat |
| 5    | chatgpt-on-wechat  | 支持多种平台接入的聊天机器人，支持多种语言模型。             | https://github.com/zhayujie/chatgpt-on-wechat        |
| 6    | anything-llm       | 一款全栈应用程序，可让您将任何文档、资源或内容转换为上下文   | https://github.com/Mintplex-Labs/anything-llm        |
| 7    | FastGPT            | 一个基于LLM的知识平台，提供数据加工、RAG检索和视觉AI工作流编排等能力。 | https://github.com/labring/FastGPT                   |
| 8    | RAGFlow            | 基于深度文档理解的开源 RAG（检索增强生成）引擎。             | https://github.com/infiniflow/ragflow                |

**技巧3：更多**

跟踪趋势：跟踪Github上开发者、开源项目的热门趋势。可以跟踪当天、当周、当月热门趋势。网址：https://github.com/trending

中国用户可以关注推荐Github项目的国内公众号，我日常关注的有：逛逛Github、HelloGitHub、GitHubDAIly。

善用Awesome：Awesome集合了某个领域优势信息，信息质量极高。比如以下几个：

| 序号 | 名称                            | 简介                                                    | Github仓库地址                                               |
| ---- | ------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| 1    | awesome-chatgpt                 | chatgpt优质资源合集，包括论文、AI相关公司、学习视频等。 | https://github.com/OpenMindClub/awesome-chatgpt              |
| 2    | Awesome-LLM                     | LLM精选列表，涵盖常见LLM介绍、LLM工具、LLM相关课程。    | https://github.com/Hannibal046/Awesome-LLM                   |
| 3    | Awesome-LLMs-Datasets           | 总结了现有最具有代表性LLMs文本数据集，非常全面。        | https://github.com/lmmlzn/Awesome-LLMs-Datasets              |
| 4    | awesome-generative-ai-guide     | 生成式AI相关最新研究、访谈等材料。                      | https://github.com/aishwaryanr/awesome-generative-ai-guide   |
| 5    | Awesome-Diffusion-Models        | Diffusion模型有关的资源和论文。                         | https://github.com/diff-usion/Awesome-Diffusion-Models       |
| 6    | awesome-artificial-intelligence | 集合AI相关的视频、书籍、讲座和论文等。                  | https://github.com/owainlewis/awesome-artificial-intelligence |
| 7    | Awesome ChatGPT Prompts         | 提示词合集。                                            | https://github.com/f/awesome-chatgpt-prompts                 |

##### Hugging Face：AI从业者都在这里

网址：https://huggingface.co

Hugging Face 的核心优势在于其 Transformers 库，提供了大量预训练的自然语言处理模型，使开发者能够轻松实现任务如文本分类、问答系统和翻译等，并且通过简单的 API 加速应用开发，同时拥有强大的社区支持和持续更新的模型资源。

下面介绍Hugging Face的三个使用技巧：

微调预训练模型：Hugging Face 的 Transformers 库包含大量的预训练大模型（如 BERT、GPT-3 等），这些模型可以通过少量数据进行微调，以适应特定任务。这种方法不仅节省计算资源，还能迅速获得高性能的结果。

利用社区共享资源：Hugging Face Hub 是一个开放的平台，用户可以上传并分享自己的模型和数据集。您可以直接从 Hub 下载并使用他人已经优化的大模型或经过清洗的数据集，这样可以加速开发过程，并且经常能找到针对特定领域或语言优化过的版本。

集成 Inference API 实现实时推理：通过使用 Hugging Face 的 Inference API，可以轻松将大模型集成到应用中，实现实时推理服务。这对于需要快速部署而不想管理复杂基础设施的项目特别有用，因为它提供了即插即用的解决方案。

##### Arxiv：获取AI前沿研究进展

网址：https://arxiv.org

ArXiv 是一个开放获取的学术论文平台，提供最新 AI 大模型研究，包括 GPT-3 和 BERT 等经典论文，是获取源头信息的首选。1991年8月至今，arXiv的总提交数量达到了235万篇。

2023年10月，arXiv达到了20,710篇新提交的月度记录，其中计算机科学、数学和物理学领域贡献了超过15,000篇。

作为非专业研究人员，并且时间有限，推荐你使用以下方法跟踪AI前沿进展：

- 活水快报。由活水智能开发，每天精选12篇AI相关的论文，网址：42digest.io，公众号「AI 研究速递」。
- 订阅关键词、查看顶级期刊筛选的论文。关键词：LLM or GenAI or AI site:arxiv.org、LLM site:nature.com、LLM site:scientificamerican.com 、LLM site:science.org、LLMsite:sciencedAIly.com、LLM site:cell.com
- Aminer筛选的论文。信息很全面，还可用于跟踪AI领域顶尖学者、研究机构等信息。网址：https://www.aminer.cn/topic

以上三个信息源，是跟踪AI大模型前言动态的首选。

#### 重要公司

除此之外，还可以跟踪顶尖AI公司的博客、阅读技术手册、阅读官方教程。博客代表了他们最新动态，非常值得关注。

我们按照AI大模型产业链整理了最值得关注的一批公司。整个产业链包括信息预处理、信息预训练、信息生成、信息微调与校验、信息的二次分发与加工、从信息世界到物理世界。

判断标准是这些公司在行业里面足够有影响力、已经盈利（包括有影响力但是没有盈利的）。

##### 数据预处理

（1）数据获取

- CrowdFlower (现为 [Figure Eight](https://www.figure-eight.com/) )：提供数据标注和清洗服务。
- [Kaggle Datasets](https://www.kaggle.com/datasets)：提供各种公开数据集。
- 八友科技：中国大模型公司头部数据提供商。

（2）数据标注

- [Appen](https://appen.com/)：全球领先的数据标注和人工智能训练数据提供商。
- [Scale AI](https://scale.com/)：为自动驾驶、计算机视觉等领域提供高质量的数据标注服务。
- [Labelbox](https://labelbox.com/)：一个用于图像、文本和视频标注的平台，支持机器学习模型的训练。

##### 大模型预训练

- [Open AI](https://openAI.com/)：代表大模型GPT-3.5，GPT-4，GPT-4o。

- [Anthropic](https://www.anthropic.com/)。代表大模型Claude 3、Claude 3.5 Sonnet。

##### 中间服务提供商

（1）向量数据库

- [MyScale](https://myscale.com/)：是一种新兴但有潜力的大规模分布式存储与计算平台，它在处理高维度数据方面表现出色，并且正在逐渐获得更多关注。
- [Pinecone](https://www.pinecone.io/)：一个完全托管的向量数据库，专为机器学习和AI应用设计，提供快速、高效的相似性搜索。
- [Milvus](https://milvus.io/)：一个开源向量数据库，支持大规模、高性能的相似性搜索和分析。它由Zilliz公司开发，并被广泛用于各种AI应用中。
- [Weaviate](https://weaviate.io/)：一个开源分布式向量搜索引擎，可以用于构建语义搜索、推荐系统等应用。它支持多种数据类型和集成。
- [Qdrant](https://qdrant.tech/)：开源矢量（嵌入）存储引擎，可以实现高效、实时地进行近邻检索，非常适合需要低延迟查询的大型机器学习模型部署场景。

（2）算力芯片

- [英伟达](https://www.nvidia.com/)：GPU（图形处理单元）市场领导者，广泛用于深度学习模型训练。

##### 推理、微调与部署分发

- [replicate](https://replicate.com/)：提供了一种简单的方法来运行机器学习模型，无需设置复杂的基础设施，非常适合快速原型设计和实验。
- [together. AI](https://www.together.AI/) ：一个专注于协作和优化数据科学工作流程的平台。它提供了一系列工具和服务，帮助团队更高效地管理、训练和部署机器学习模型。该平台旨在简化复杂的AI开发过程，并促进团队之间的合作。
- [Anyscale](https://www.anyscale.com/)：基于Ray框架，为开发者提供了简化的大规模分布式计算环境，使得构建、训练和部署大规模AI应用变得更加容易。

##### 运用开发

基于AI大模型开发的各类运用，以下是第一批享受AI时代发展红利的运用。

（1）文生图

- Midjourney：作为AI艺术生成领域的先锋，Midjourney以其高质量、艺术风格的图像生成能力在创意社区中广受欢迎，成为数字艺术家和设计师的首选工具，在市场上占据重要位置。网址：https://www.midjourney.com/home
- Stability AI：凭借其开源项目Stable Diffusion，Stability AI 在文本到图像生成领域树立了标杆，被广泛应用于学术研究和商业项目，是行业内公认的技术领导者，其技术被许多企业采用。网址：https://stability.AI/
- Civit AI ：通过提供一个社区驱动的平台，Civit AI 成为用户分享和探索多样化AI生成艺术作品的重要枢纽，在创意交流中占据重要位置，并推动了用户参与度和创新力。网址：https://civitAI.com/

（2）文生音乐

- suno：利用先进的AI技术，将文字描述转换为情感丰富且个性化的音乐作品，适用于各种创意项目。网址：https://suno.com/

（3）文生视频：

- runway：提供强大的机器学习工具，使用户能够通过简单的文本输入快速创建复杂且专业的视频内容，包括动画和特效。网址：https://runwayml.com/
- 可灵：可灵图生视频模型以卓越的图像理解能力为基础，将静态图像转化为生动的5秒精彩视频。配上创作者不同的文本输入，即生成多种多样的运动效果，让您的视觉创意无限延展。网址：https://klingAI.kuAIshou.com/