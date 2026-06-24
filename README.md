# PHADCCD-Net-Cross-Lingual-Multimodal-Hate-Speech-Detection-via-Concordance-Discordance-Decomposition

This is the official implementation of "PHADCCD-Net-Cross-Lingual-Multimodal-Hate-Speech-Detection-via-Concordance-Discordance-Decomposition".

Awarded 3rd overall and accepted for publication at [IEEE ICME 2026](https://2026.ieeeicme.org/) Conference under the ECHO Grand Challenge Track.

### Audio-only track

<p align="justify"> PHAD-Net for the audio-track is a prosody-aware neural architecture for multilingual hate/aggression audio detection. It targets prosodic cues—pitch, intensity, speech rate, and voice quality that operate at both frame and utterance time scales. The model has three branches. First, a Wav2Vec2-XLS-R-300M encoder (pretrained on 128 languages) provides frame-level representations; its feature extractor and bottom 20 transformer layers are frozen, and the top 4 layers are fine-tuned. Frame outputs are aggregated with learned attention pooling, so high-salience frames (e.g., intensity peaks, pitch breaks, voiced–unvoiced transitions) are emphasized instead of being washed out by mean pooling. Second, a Prosodic Dynamics Network (PDN) processes four contour sequences: F0, RMS energy, spectral centroid, and zero-crossing rate—computed over 25 ms frames. A lightweight 1D-CNN stack with attention pooling captures temporal prosodic patterns such as rising pitch and intensity bursts that XLS-R is not explicitly biased to learn. Third, a handcrafted branch encodes prosodic statistics, voice quality measures, and spectral descriptors (MFCCs and related features) via a two-layer MLP, providing robust summaries for noisy or very short utterances. The three branches are fused using a gated mechanism conditioned on a per-language embedding, allowing the model to up/down-weight each branch dynamically per sample, and the fused embedding is fed to a binary classifier. Training uses focal loss with label smoothing to handle class imbalance, a language-balanced sampler for equal coverage of 10 languages, gradient accumulation to reach an effective batch size of 36, and differential learning rates with cosine annealing (lower for the pretrained encoder, higher for custom heads). This design captures complementary acoustic abstractions and improves robustness to language diversity and variable recording conditions. </p>

### Overall workflow:

<img width="631" height="451" alt="PHADAudio drawio (1)" src="https://github.com/user-attachments/assets/6f85ce6f-ee5e-462c-a845-d468442eea87" />

### Text-only track

<p align="justify"> LCTS-Net (Language-Conditioned Token Saliency Network) for the text-only track is a multilingual hate speech detection model that explicitly adapts its notion of “important tokens” to each language. Built on XLM-RoBERTa-base with lower layers frozen, it uses three complementary branches. Branch A is a stabilized [CLS]-based sentence representation that serves as a reference. Branch B introduces language-conditioned attention: a learned language embedding acts as a query over token hidden states, producing language-specific salience patterns (e.g., focusing on context-dependent abuse in Hindi, euphemisms in Chinese, or code-switch boundaries in Bengali-English). Branch C is an anchor detector that learns a sparse distribution over tokens via an auxiliary entropy penalty, encouraging the model to concentrate on the few words that typically signal hate in otherwise neutral text and yielding interpretable anchor weights. The three branch outputs, concatenated with the language embedding, feed a small classifier. Training uses focal loss with label smoothing, language-balanced sampling, and differential learning rates, plus the sparsity regularizer. This design allows LCTS-Net to share a single classifier across languages while learning language-specific token saliency, improving robustness and interpretability for hate detection in typologically diverse and code-mixed text.</p>

### Overall workflow:

<img width="701" height="551" alt="LCTSText drawio (1)" src="https://github.com/user-attachments/assets/b9198d56-f863-4eaf-aacd-6348f9ae6608" />

### Audio + Text track

<p align="justify">PHADCCD-Net (Prosodic Hate Audio Detection with ConCordance–discordance Decomposition Network)for the audio+text track is a multimodal architecture for hate speech detection that focuses not on fusion per se, but on explicitly modeling when audio and text agree or conflict. The audio branch reuses the XLS-R-300M encoder and prosodic dynamics network from the unimodal setup, while the text branch uses XLM-RoBERTa-base with lower layers frozen. Before fusion, Cross-Modal Temporal Attention lets audio frames attend to text tokens and vice versa, so each modality is contextualized by the other (e.g., shouted segments aligned with either slurs or celebratory phrases). The resulting representations pass through a shared–private encoder structure: modality-specific private MLPs capture audio texture and textual syntax, while a shared MLP maps both into a common semantic space, with an orthogonality penalty preventing the shared and private subspaces from collapsing onto each other.

ConCordance–discordance Decomposition (CCD) operates on the shared audio–text representations. Concordance is modeled as the elementwise product of the shared audio and text vectors, highlighting dimensions where the modalities align; discordance is the elementwise absolute difference, emphasizing dimensions where they diverge. Linear projections of these concordance and discordance signals are concatenated, giving the classifier explicit access to both agreement and conflict features, rather than requiring it to infer clashes from a single fused vector. A gating network produces per-branch weights, and modality dropout (p = 0.15) randomly removes entire branches during training, preventing overreliance on any single stream and improving robustness when one modality is noisy or missing. Training uses focal loss plus orthogonality regularization, gradient accumulation for an effective batch size of 48, and language-balanced sampling with differential learning rates as in the unimodal models. This design is particularly suited to cross-lingual multimodal hate speech, where calm delivery of hateful content and aggressive delivery of benign content both occur: by separating concordance and discordance, PHADCCD-Net can handle both aligned and conflicting audio–text pairs without defaulting to either modality.</p>

### Overall workflow:

<img width="799" height="721" alt="PHADAudioText drawio (1)" src="https://github.com/user-attachments/assets/4ee2bc72-3ceb-491c-9391-55713173505f" />

## Authors :

[Kushaj Mallick](https://www.linkedin.com/in/kushaj-mallick-a0926318a/)


