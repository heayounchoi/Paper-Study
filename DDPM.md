### [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

**Abstract**
- We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics.
- Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding.

---

**Introduction**
- A diffusion probabilistic model is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time.
- Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed.
- When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.
- We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models.
- we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling
- Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models.
- We find that the majority of our models' lossless codelengths are consumed to describe imperceptible image details.
- We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

---

**Background**

<img src="https://velog.velcdn.com/images/heayounchoi/post/973a51d6-d560-4db3-a86d-1c0c20a4c4a0/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/a09fb9d4-322e-4f76-bf55-383e90fe8ae2/image.png">

---

**Diffusion models and denoising autoencoders**

<img src="https://velog.velcdn.com/images/heayounchoi/post/1cf52df0-1e7f-4882-a0d9-70fdb9bb3897/image.png">

**_Forward process and $$L_T$$_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/bfeecdc3-e594-42e1-b06f-59850c203ec9/image.png">

**_Reverse process and_** $$L_{1:T-1}$$

<img src="https://velog.velcdn.com/images/heayounchoi/post/7726e3ad-9884-48a1-9389-b5b60fa95249/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/7006fee7-b884-4794-8586-fac89ca5ca3f/image.png">

**_Data scaling, reverse process decoder, and $$L_0$$_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/a41a02df-3df4-454a-909b-568aeb8a0cf9/image.png">

**_Simplified training objective_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/bffd236e-4796-4d16-8513-d41ba1bde79e/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/a86f7787-c766-4eea-a257-290db583b3c9/image.png">

---

**Experiments**

<img src="https://velog.velcdn.com/images/heayounchoi/post/abf3fd06-943e-425e-acd9-9e20bd02645b/image.png">

**_Sample quality_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/8b0879d3-d65a-464f-a49c-6a81e5207160/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/dc94d7fd-19ae-4944-84a6-8dec975c664f/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/7aaf95b9-7abf-4434-80e2-3029ccea2027/image.png">

**_Reverse process parameterization and training objective ablation_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/933e9f0d-c9e3-4aff-b211-dc8641215eee/image.png">

**_Progressive coding_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/687831e8-3af2-485d-8f8b-09fb5bee46f7/image.png">

_Progressive lossy compression_

<img src="https://velog.velcdn.com/images/heayounchoi/post/4e6d89f0-d055-42cb-a6b8-1d7bedfff488/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/e8f16ec3-588f-4f58-a14f-a65b5715294d/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/6703d6b7-9bf1-410e-9edb-46dab61b848b/image.png">

_Progressive generation_

<img src="https://velog.velcdn.com/images/heayounchoi/post/16f9754c-e999-45fa-bfdd-c494328f5c77/image.png">

_Connection to autoregressive decoding_

<img src="https://velog.velcdn.com/images/heayounchoi/post/09f2d3fa-e4dc-4260-b80a-d8666c24f345/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/b12ea462-6c37-4e12-9618-dbd13720d2a4/image.png">

**_Interpolation_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/ef69cdba-4d16-4b90-b3e1-7d87dc0686ca/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/534fc1f4-dd4c-4eab-ad73-068096fe57a7/image.png">

---

**Related Work**

---

**Conclusion**

---

**Broader Impact**
