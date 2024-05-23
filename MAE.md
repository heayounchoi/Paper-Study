### [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377)

**Abstract**
- MAE are scalable self-supervised learners for computer vision
- mask random patches of the input image and reconstruct the missing pixels
- two core designs:
(1) an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens
(2) masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task
- Coupling these two designs enables to train large models efficiently and effectively: accelerates training (by 3x or more) and improves accuracy.
- allows for learning high-capacity models that generalize well
- Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.
---

**Introduction**
- progress of autoencoding methods in vision lags behind NLP
- Convolutions typically operate on regular grids and it is not straightforward to integrate 'indicators' such as mask tokens or positional embeddings into convolutional networks.
- after ViT, no longer an obstacle
- information density is different between language and vision
- languages are human-generated signals that are highly semantic and information-dense
- images are natural signals with heavy spatial redundancy--e.g., a missing patch can be recovered from neighboring patches with little high-level understanding of parts, objects, and scenes
- masking a very high portion of random patches largely reduces redundancy and creates a challenging self-supervisory task that requires holistic understanding beyond low-level image statistics

<img src="https://velog.velcdn.com/images/heayounchoi/post/8bd6b02c-17f8-409e-af96-cc3fb5de4d56/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/963067ef-61ec-4d8d-bef7-636d22a62a02/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/23d79d4a-2008-4dfc-8b7f-4e8c6c8f0234/image.png">

- The autoencoder's decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images.
- In vision, the decoder reconstructs pixels, hence its output is of a lower semantic level than common recognition tasks.
- In language, the decoder predicts missing words that contain rich semantic information.
- For images, the decoder design plays a key role in determining the semantic level of the learned latent representations.

<img src="https://velog.velcdn.com/images/heayounchoi/post/bef966f2-cd7c-448a-ba5c-9325803efa9b/image.png">

- Shifting the mask tokens to the small decoder in the asymmetric encoder-decoder results in a large reduction in computation.
- This can reduce overall pre-training time by 3x or more and likewise reduce memory consumption, enabling to easily scale MAE to large models.
- With MAE pre-training, we can train data-hungry models with improved generalization performance.
---

**Related Work**

_Masked language modeling_
- BERT, GPT

_Autoencoding_
- learns representations
- It has an encoder that maps an input to a latent representation and a decoder that reconstructs the input.
- PCA, k-means, denoising autoencoders (DAE)

_Masked image encoding_
- Context Encoder, iGPT, ViT, BEiT

_Self-suprevised learning_
- contrastive learning with data augmentation
---

**Approach**
- Unlike classical autoencoders, MAE has an asymmetric design that allows the encoder to operate only on the partial, observed signal (without mask tokens) and a lightweight decoder that reconstructs the full signal from the latent representation and mask tokens.

_Masking_
- random sampling (uniform distribution)
- Random sampling with a high masking ratio largely eliminates redundancy, thus creating a task that cannot be easily solved by extrapolation from visible neighboring patches.

_MAE encoder_
- a ViT but applied only on visible, unmasked patches
- masked patches are removed; no mask tokens are used

_MAE decoder_
- The input to the MAE decoder is the full set of tokens consisting of (i) encoded visible patches, and (ii) mask tokens.
- Each mask token is a shared, learned vector that indicates the presence of a missing patch to be predicted.
- The MAE decoder is only used during pre-training to perform the image recognition task.
- default decoder has <10% computation per token vs. the encoder

_Reconstruction target_
- MAE reconstructs the input by predicting the pixel values for each masked patch
- loss function computes the MSE between the reconstructed and original images in the pixel space
- compute the loss only on masked patches, similar to BERT
- traditional denoising autoencoders compute the loss on all pixels
- using normalized pixels as the reconstruction target improves representation quality

_Simple implementation_
1) generate token for every input patch
2) randomly shuffle the list of tokens and remove the last portion of the list based on the masking ratio
3) encoder
4) append a list of mask tokens to the list of encoded patches and unshuffle this full list to align all tokens with their targets
5) decoder
---

**ImageNet Experiments**

_Baseline: ViT-Large_
- ViT-L is very big and tends to overfit
- strong regularization is needed

<img src="https://velog.velcdn.com/images/heayounchoi/post/6a215dee-b67e-4048-8a83-f6289779db19/image.png">

- fine-tuning accuracy heavily depends on pre-training

**_Main Properties_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/a11a21ff-c526-4ee5-b804-1d7f80c14303/image.png">

_Masking ratio_

<img src="https://velog.velcdn.com/images/heayounchoi/post/589a329c-0e6f-469f-917b-ee0f2e3ac557/image.png">

- BERT's typical masking ratio is 15%
- for fine-tuning, the results are less sensitive to the masking ratios

_Decoder design_
- only has 9% FLOPs per token vs ViT-L

_Mask token_

<img src="https://velog.velcdn.com/images/heayounchoi/post/5b5c26dc-7e51-4785-9870-18a81dc0450a/image.png">

- If the encoder uses mask tokens, it performs worse.
- encoder has a large portion of mask tokens in its input in pre-training, which does not exist in uncorrupted images
- this gap may degrade accuracy in deployment
- by removing the mask token from the encoder, we constrain the encoder to always see real patches and thus improve accuracy
- enables large-batch training

_Reconstruction target_
- per-patch normalization enhances the contrast locally
- pixel-based MAE is much simpler than tokenization
- The dVAE tokenizer requires one more pre-training stage, which may depend on extra data.
- The dVAE encoder is a large convolutional network (40% FLOPs of ViT-L) and adds nontrivial overhead.

_Data augmentation_
- In MAE, the role of data augmentation is mainly performed by random masking.
- The masks are different for each iteration and so they generate new training samples regardless of data augmentation.

_Mask sampling strategy_

<img src="https://velog.velcdn.com/images/heayounchoi/post/5a173ce8-ad75-46eb-bf2f-067ae5085b05/image.png">

_Training schedule_

<img src="https://velog.velcdn.com/images/heayounchoi/post/02f5ec93-9517-4760-8e57-29cc6c1b6741/image.png">

- contrastive learning methods saturates at 300 epochs for ViT-L
- MAE encoder only sees 25% of patches per epoch, while in constrastive learning the encoder sees 200% or even more patches per epoch

**_Comparisons with Previous Results_**

_Comparisons with self-supervised methods_

<img src="https://velog.velcdn.com/images/heayounchoi/post/541bfacd-5125-4999-8459-5ddad35fba3d/image.png">

- For ViT-L, the gaps among methods are bigger, suggesting that a challenge for bigger models is to reduce overfitting.
- BEiT reported degradation when reconstructing pixels with ViT-B

_Comparisons with supervised pre-training_

<img src="https://velog.velcdn.com/images/heayounchoi/post/20dc1f11-71c3-4375-9c6d-ca5dd703fd04/image.png">

- the gain over training from scratch is bigger for higher-capacity models

**_Partial Fine-tuning_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/bace8f08-41bd-4428-9f39-a62950f9fb64/image.png">

- fine-tuning a few blocks can achieve accuracy close to full fine-tuning
- While the MAE representations are less linearly separable, they are stronger non-linear features and perform well when a non-linear head is tuned.
- These observations suggest that linear separability is not the sole metric for evaluating representation quality.
- It has also been observed that linear probing is not well correlated with transfering learning performance.
---

**Transfer Learning Experiments**

_Object detection and segmentation_

<img src="https://velog.velcdn.com/images/heayounchoi/post/93884f5e-c771-44b4-8eab-26210e24eb10/image.png">

_Semantic segmentation_

<img src="https://velog.velcdn.com/images/heayounchoi/post/ec2e6563-562e-4812-9ee6-558feb5c9ab6/image.png">

_Classification tasks_

<img src="https://velog.velcdn.com/images/heayounchoi/post/4744002b-95a5-43f7-b5b2-99b97a9bc6d0/image.png">

_Pixels vs. tokens_

<img src="https://velog.velcdn.com/images/heayounchoi/post/3a37db22-66da-487b-b5fa-854eea2fc292/image.png">

---

**Discussion and Conclusion**

_Broader impacts_

