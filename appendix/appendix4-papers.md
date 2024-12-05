# 备查表4：经典论文

AI大模型技术的发展中起到了至关重要的作用，还奠定了当前许多自然语言处理应用的基础，每一篇论文都代表了一个重要的里程碑。

1. Attention is All You Need (2017) - Vaswani et al.
   - 简介：提出了Transformer架构，通过自注意力机制解决了传统RNN和CNN的局限性。
   - [论文链接](https://arxiv.org/abs/1706.03762)
   
2. Deep contextualized word representations (2018) - Peters et al.
   - 简介：引入了基于双向LSTM的ELMo模型，通过上下文进行动态词向量表示，极大地提升了下游NLP任务的表现。
   - [论文链接](https://arxiv.org/abs/1802.05365)
   
3. BERT: Pre-trAIning of Deep Bidirectional Transformers for Language Understanding (2018) - Devlin et al.
   - 简介：提出了BERT模型，通过双向Transformer的预训练方式显著提高了多种自然语言处理任务的效果。
   - [论文链接](https://arxiv.org/abs/1810.04805)
   
4. Improving Language Understanding by Generative Pre-TrAIning (2018) - Radford et al.
   - 简介：提出了生成式预训练模型GPT，通过无监督预训练和有监督微调提升语言理解能力，开创了生成式预训练模型的先河。
   - [论文链接](https://openAI.com/research/language-unsupervised)
   
5. RoBERTa: A Robustly Optimized BERT PretrAIning Approach (2019) - Liu et al.
   - 简介：对BERT预训练方法进行了优化，通过更大的训练数据和更长的训练时间显著提高了模型性能。
   - [论文链接](https://arxiv.org/abs/1907.11692)
   
6. FAISS: A Library for Efficient Similarity Search and Clustering of Dense Vectors (2017) - Johnson et al.
   - 简介：向量数据库方面经典论文。FAISS是一个高效的相似性搜索和密集向量聚类的库，它被广泛用于在大规模数据集中快速检索向量。
   - [论文链接](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
   
7. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019) - Raffel et al.
   - 简介：提出了T5模型，统一了文本生成和文本理解任务，通过“文本到文本”的转换方式处理各种NLP任务。
   - [论文链接](https://arxiv.org/abs/1910.10683)
   
8. GPT-2: Language Models are Unsupervised Multitask Learners (2019) - Radford et al.
   - 简介：提出了GPT-2模型，通过无监督学习实现了多任务处理，显著提升了语言生成的质量。
   - [论文链接](https://openAI.com/research/language-unsupervised)
   
9. Scaling Laws for Neural Language Models (2020) - Kaplan et al.
   - 简介：探讨了神经语言模型的规模与性能之间的关系，为大规模模型的设计和训练提供了理论依据。
   - [论文链接](https://arxiv.org/abs/2001.08361)
   
11. Electra: Pre-trAIning Text Encoders as Discriminators Rather Than Generators (2020) - Clark et al.
    - 简介：提出了Electra模型，通过预训练判别器而非生成器的方法，提高了预训练的效率和效果。
    - [论文链接](https://arxiv.org/abs/2003.10555)
    
12. Language Models are Few-Shot Learners (2020) - Brown et al.
    - 简介：展示了GPT-3模型，通过庞大的参数量和无监督预训练，在少量示例下实现了卓越的任务表现。
    - [论文链接](https://arxiv.org/abs/2005.14165)
    
13. What does BERT look at? An Analysis of BERT's Attention(2019) - Clark et al.
    - 简介：深入分析了BERT模型内部的工作机制，揭示了其内部表示和注意力模式的特点。
    - [论文链接](https://arxiv.org/abs/1906.04341)
    
14. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (2019) - Sanh et al.
    - 简介：提出了DistilBERT模型，通过知识蒸馏技术减小了BERT模型的大小，同时保持了较高的性能。
    - [论文链接](https://arxiv.org/abs/1910.01108)
    
15. Retrieval-Augmented Generation for Language Models (2020) - Lewis et al.
    - 简介：提出了RAG模型，通过结合检索技术和生成模型，增强了模型在回答问题时的信息准确性和丰富性。
    - [论文链接](https://arxiv.org/abs/2005.11401)
    
16. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2021) - Fedus et al.
    - 简介：通过引入稀疏激活技术实现了高效的超大规模模型训练，为大规模模型的训练提供了一种新的方法。
    - [论文链接](https://arxiv.org/abs/2101.03961)
    
17. Perceiver: General Perception with Iterative Attention (2021) - Jaegle et al.
    - 简介：提出了Perceiver模型，通过迭代注意力机制处理任意类型的数据输入，为构建通用感知模型提供了新思路。
    - [论文链接](https://arxiv.org/abs/2103.03206)
    
19. Training Compute-Optimal Large Language Models(2022) - Hoffmann et al.
    - 简介：研究了大规模模型的训练效率，提出了在固定计算预算下，更多数据和较小模型的训练策略，为训练大规模模型提供了新思路。
    - [论文链接](https://arxiv.org/abs/2203.15556)
    
20. PaLM: Scaling Language Modeling with Pathways (2022) - Chowdhery et al.
    - 简介：展示了PaLM模型如何利用Pathways架构实现多任务和多模态的高效训练，推动了大规模多模态模型的发展。
    - [论文链接](https://arxiv.org/abs/2204.02311)
    
21. Scaling Language Models: Methods, Analysis & Insights from Training Gopher (2022) - Rae et al.
    - 简介：详细探讨了大规模语言模型的训练方法和扩展策略，通过对280B参数模型的训练过程进行深入分析，为理解和改进大型语言模型的性能提供了重要见解。
    - [论文链接](https://arxiv.org/abs/2112.11446)
    
22. GPT-4 Technical Report (2023) - OpenAI
    - 简介：报告了GPT-4的开发，作为多模态大规模模型，它在多个专业和学术基准测试中表现出人类级别的性能，是GPT系列的最新发展。
    - [论文链接](https://cdn.openAI.com/papers/gpt-4.pdf)
    
23. LLaMA: Open and Accurate Large Language Models (2023) - Meta AI
    - 简介：介绍了LLaMA模型，这是一个开源的大规模语言模型，旨在提高模型的准确性和透明度。
    - [论文链接](https://arxiv.org/abs/2302.13971)

###