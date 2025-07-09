# Song_MixingStructure_Segment

## Overview

This is a part of my bachelor thesis project at University of Virginia. The primary goal of this song segmentation task is to **detect temporal changes in mixing characteristics**, rather than simply identifying traditional musical structures. While common structures like verses, choruses, intros, and outros often correlate with changes in mixing, the focus here is on capturing **shifts in spectral characteristics** and **mixed features over time** ([McFee et al., 2014](#references)).

For instance, **choruses** typically serve as the song’s climax and feature more layers of instrumentation, broader frequency content, and higher loudness. In contrast, **intros** and **outros** tend to be minimal with lower energy levels. Thus, segmenting based only on loudness-related features (e.g., RMS or energy) may miss these subtle transitions.

## Feature Design

To address this, I incorporated additional **frequency-domain features**, including:

- **Spectral Flux**: Captures changes in spectral content over time.
- **MFCCs (Mel-Frequency Cepstral Coefficients)**: Represents the **timbral texture** of the audio.

These features help detect subtle mixing changes and support more accurate, context-aware segmentation.

## Segmentation Pipeline

The song segmentation process follows **four main stages**:

1. **Feature Extraction**
2. **Temporal Aggregation**
3. **Change Point Detection**
4. **Structural Mapping**

### 1. Feature Extraction

I use a **sliding window** (size = 2048, hop length = 512) to extract short-time features:

- Energy
- Root Mean Square (RMS) Amplitude
- Spectral Flux
- MFCCs

These features are normalized and combined into a **composite signal** `x_t` to represent the mix:

x_t = 0.2 * energy_t + RMS_t + spectral_flux_t + MFCC_t


### 2. Temporal Aggregation

To capture broader trends without losing transients like kick drum impacts:

- A **1-second moving average** filter is applied to `x_t`.
- The end of the signal is **zero-padded** to ensure the final second is processed, causing the feature to converge toward 0.

This produces a smooth signal suitable for structural segmentation.

### 3. Change Point Detection

I use the **PELT (Pruned Exact Linear Time)** algorithm ([Killick et al., 2012](#references)) via the [`ruptures`](https://github.com/deepcharles/ruptures) library with an **RBF kernel**, which helps detect **non-linear feature changes**.

### 4. Structural Mapping

To align corresponding sections across tracks, I simplify the structure into a **fixed 4-to-5 phase framework**:

Intro → Drop1 → Bridge → Drop2 → Outro


- **Intro**: Low energy, limited frequency content.
- **Drop1**: First climax (chorus), with fuller mix and energy.
- **Bridge**: A transitional phase preparing for Drop2.
- **Drop2**: Second climax.
- **Outro** (optional): If present in both tracks, included explicitly; otherwise, merged with Drop2.

### Label Merging Strategy

Since change point detection can produce fine-grained segments, I use a **threshold-based merging strategy** to combine adjacent segments based on:

- Temporal order
- Feature values (energy, spectral flux, MFCCs)

This allows simplified, interpretable section labeling.

## Output

The final segmentation:

- Annotates each **section boundary and duration**
- Enables **visual and analytical comparison** across tracks

## Evaluation on Song Segmentation

### Example 1: *Wake Me Up* by Avicii

Firstly, I evaluated whether the song segmentation method correctly identifies distinct sections. I used a well-known electronic dance music track, *Wake Me Up* by Avicii, as an example. This song follows a very standardized structure:

Intro → Pre-Chorus → Chorus → Bridge → Pre-Chorus → Chorus


Each section shows different spectral and loudness characteristics. The song also includes diverse instrumentation, ranging from acoustic to electronic instruments, making it an ideal test case.

#### Spectrogram Analysis

The figure below shows the spectrogram of *Wake Me Up*. From the spectrogram, different sections exhibit distinct frequency distributions and loudness profiles.

![Wake_Me_Up_spectrogram](https://github.com/user-attachments/assets/6896e31f-b0bf-4bf5-8468-b2f1cf04362e)


#### Feature Detection

The feature detection algorithm identifies nine distinct sections. These align well with the spectrogram patterns. For example, the algorithm correctly isolates the intro (0s–39s) and the pre-chorus (39s–68s), providing a strong foundation for further reduction to mixing-based segmentation.

![Wake_Me_Up_change_points](https://github.com/user-attachments/assets/9e26885c-a7d8-4414-a0e5-ca60a559ee0f)

#### Segment Labeling

In the final segmentation result, adjacent segments were merged based on feature similarity and structure. For instance, the segment from 39s to 113s is labeled as `Drop1`. While the section from 68s to 86s serves as a buildup (lower power), it's still grouped into `Drop1` due to its relative energy and proximity.

A similar merge is done for 181s to 210s into `Drop2`. A 2-second overlap is applied at boundaries to ensure smoother transitions. As a result, the track is segmented into four coherent regions that reflect both musical structure and perceptual mixing changes.

![Wake_Me_Up_song_structure](https://github.com/user-attachments/assets/99675530-58b1-45cf-999e-a6cf1ebe72a8)

---

### Example 2: *Toxic* by Britney Spears

Unlike *Wake Me Up*, *Toxic* by Britney Spears features a more unconventional mixing structure. According to the [CCMusic Database](https://github.com/rockyzhengwu/ccmusic-database) ([Li et al., 2019](#references)), the song is divided into 13 traditional pop sections:

#### Official Structure Table

| Section # | Start Time (s) | End Time (s) | Structure Annotation |
|-----------|----------------|--------------|-----------------------|
| 1         | 00.00          | 42.41        | Intro                 |
| 2         | 42.41          | 69.24        | Verse A               |
| 3         | 69.24          | 86.06        | Pre-chorus A          |
| 4         | 86.06          | 112.89       | Chorus A              |
| 5         | 112.89         | 126.31       | Re-intro A            |
| 6         | 126.31         | 139.77       | Verse B               |
| 7         | 139.77         | 156.55       | Pre-chorus B          |
| 8         | 156.55         | 196.81       | Chorus B              |
| 9         | 196.81         | 240.43       | Re-intro B            |
| 10        | 240.43         | 267.30       | Chorus C              |
| 11        | 267.30         | 280.72       | Bridge A              |
| 12        | 280.72         | 294.17       | Re-intro C            |
| 13        | 294.17         | 334.43       | Chorus D              |

#### Spectrogram Analysis

Despite this traditional annotation, the mixing remains largely uniform throughout. The spectrogram confirms this homogeneity.

![Toxic_spectrogram](https://github.com/user-attachments/assets/caa43047-f99b-4ee6-a5ea-61c9a7767aa0)


#### Feature Detection

The detection result highlights this observation. While annotations list the `Intro` as 0–42s, the spectral profile changes significantly after 15s. The segment from 15s–52s shows strong similarity in spectral features and is grouped together.

Overall, the algorithm classifies the entire song into **4 mixing-based sections**. This matches the actual production characteristics rather than formal song structure.

![Toxic_change_points](https://github.com/user-attachments/assets/7ca70a66-e12d-4c6a-b813-783d5a7a992b)

#### Segment Labeling

The segmentation output uses the 4 detected regions directly. Although the official song structure has 13 annotated parts, the mixing analysis simplifies it effectively without loss of meaningful interpretation.

One limitation is that the feature extraction uses a 1-second window, limiting precision to full seconds. This restricts granularity compared to frame-level or millisecond-level annotations.

![Toxic_song_structure](https://github.com/user-attachments/assets/0a8be687-8e01-44b3-bdf4-20e1086c698d)



## References

- McFee, B., & Ellis, D. P. (2014). *Analyzing Song Structure with Spectral Clustering*. In ISMIR.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal detection of changepoints with a linear computational cost*. Journal of the American Statistical Association, 107(500), 1590–1598.
