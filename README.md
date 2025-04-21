

<div align="center">
<h2><a href="https://arxiv.org/abs/2501.18492">	
GuardReasoner: Towards Reasoning-based LLM Safeguards</a></h2>
    

</a></h2>
[Yue Liu](https://yueliu1999.github.io/), Hongcheng Gao, Shengfang Zhai, Jun Xia, Tianyi Wu, Zhiwei Xue, Yulin Chen, Kenji Kawaguchi, Jiaheng Zhang, Bryan Hooi

</div>


    

<p align = "justify">
As LLMs increasingly impact safety-critical applications, ensuring their safety using guardrails remains a key challenge. This paper proposes GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to reason. Concretely, we first create the GuardReasonerTrain dataset, which consists of 127K samples with 460K detailed reasoning steps. Then, we introduce reasoning SFT to unlock the reasoning capability of guard models. In addition, we present hard sample DPO to further strengthen their reasoning ability. In this manner, GuardReasoner achieves better performance, explainability, and generalizability. Extensive experiments and analyses on 13 benchmarks of 3 guardrail tasks demonstrate its superiority. Remarkably, GuardReasoner 8B surpasses GPT-4o+CoT by 5.74% and LLaMA Guard 3 8B by 20.84% F1 score on average. We release the training data, code, and models with different scales (1B, 3B, 8B) of GuardReasoner.
</p>
<div align="center">
    <img src=./assets/overview.png width="100%">
</div>







## Update

- (**2025/03/06**) The paper has been accepted by the ICLR 2025 FM-Wild Workshop.
- (**2025/02/02**) The [training pipeline](./train/README.md) is released.
- (**2025/02/01**) The training data [GuardReasonerTrain](https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain) is released.
- (**2025/01/31**) The models are released ([1B](https://huggingface.co/yueliu1999/GuardReasoner-1B), [3B](https://huggingface.co/yueliu1999/GuardReasoner-3B), [8B](https://huggingface.co/yueliu1999/GuardReasoner-8B)).
- (**2025/01/31**) The code of GuardReasoner is released.
- (**2025/01/31**) GuardReasoner is on [arXiv](https://arxiv.org/abs/2501.18492).







## Usage

### Quick Start
To evaluate GuardReasoner, run the following code.

```python
python ./evaluate.py
```







### Main Result

<p align="center">
Table 1: Performance on Prompt Harmfulness Detection Task.
</p>
<div align="center">
    <img src=./assets/prompt.png width="75%">
</div>







<p align="center">
Table 2: Performance of Response Harmfulness Detection Task.
</p>

<div align="center">
    <img src=./assets/response.png width="70%">
</div>


<p align="center">
Table 3: Performance on Refusal Detection Task.
</p>

<div align="center">
    <img src=./assets/refusal.png width="50%">
</div>






### Development Version

To reproduce the generation process of GuardReasoner, run the following code.

1. generate via vLLM
    ```
    CUDA_VISIBLE_DEVICES=0 python generate.py
    ```
2. evaluate performance
    ```
    python evaluate.py
    ```

To use GuardReasoner, run the following code.
```
CUDA_VISIBLE_DEVICES=0 python deploy.py
```


To reproduce the training process of GuardReasoner, see [training pipeline](./train/README.md).


## Acknowledgement
Our method are partly based on the following resources. Thanks for their awesome works.
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [WildGuard](https://github.com/allenai/wildguard)
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)



## Citations

If you find this repository helpful, please cite our paper.

```
@article{GuardReasoner,
  title={GuardReasoner: Towards Reasoning-based LLM Safeguards},
  author={Liu, Yue and Gao, Hongcheng and Zhai, Shengfang and Jun, Xia and Wu, Tianyi and Xue, Zhiwei and Chen, Yulin and Kawaguchi, Kenji and Zhang, Jiaheng and Hooi, Bryan},
  journal={arXiv preprint arXiv:2501.18492},
  year={2025}
}
```


<p align="right">(<a href="#top">back to top</a>)</p>

