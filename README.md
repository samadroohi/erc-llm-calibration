
## Supported Models

- Llama-2-7b-chat-hf
- Llama-2-13b-chat-hf
- Mistral-7B-Instruct-v0.2
- Zephyr-7b-beta

## Datasets

1. MELD
   - Emotions: neutral, surprise, fear, sadness, joy, disgust, anger

2. EmoWOZ
   - Emotions: neutral, disappointed, dissatisfied, apologetic, abusive, excited, satisfied

3. EmoCx
   - Emotions: others, happy, sad, angry

## Methods

### Verbalized Uncertainty
The verbalized approach includes multiple stages:
- Zero stage: Basic emotion prediction
- First stage: Emotion prediction with confidence estimation
- Second stage: Confidence assessment on provided predictions
- Conformal stage: Conformal prediction implementation

### Logit-based Uncertainty
Direct uncertainty estimation using model logits and softmax probabilities.

### P(true) Assessment
Evaluates the truthfulness of emotion predictions using:
- Self-assessment: Evaluating model's own predictions
- Random-assessment: Evaluating randomly assigned emotions
### Running the Code
First run the lg_uncertainty_estimation.py file to generate the uncertainty estimation for the given model and dataset.
Then put the generated data in the data folder and after updating data directory in analysis.py file run the analysis.py file to plot the results.

You can try various models and datasets by changing the model and dataset index in the lg_uncertainty_estimation.py file.

## Contributing

Please feel free to submit issues and pull requests.

## License

## License

MIT License

Copyright (c) 2024 [Samad Roohi]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

Roohi, S., Skarbez, R., Nguyen, H. (2025). Beyond Factualism: A Study of LLM Calibration Through the Lens of Conversational Emotion Recognition. In: Gong, M., Song, Y., Koh, Y.S., Xiang, W., Wang, D. (eds) AI 2024: Advances in Artificial Intelligence. AI 2024. Lecture Notes in Computer Science(), vol 15442. Springer, Singapore. https://doi.org/10.1007/978-981-96-0348-0_15
