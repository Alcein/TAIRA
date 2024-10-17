<h1>Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent</h1>


## ⚙    Preparations
Set up a virtualenv and install the [pytorch](https://pytorch.org/get-started/previous-versions/) manually. After that, install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```
Our experiments have been tested on **Python 3.10.14**.
## ⚙️ Configuration (`system_config.yaml`)

Before running the program, you need to configure the `system_config.yaml` file based on your specific requirements.


### Explanation of the Parameters:
- `QUERY_NUMBER`: Limits the size of the dataset. Here, it's set to `200`, meaning it will process up to 200 data points.
- `TOPN_ITEMS`: Number of top items returned, set to `500`.
- `TOPK_ITEMS`: The number of top K items selected, set to `10`.
- `DOMAIN`: The data domain being used. Options include `"amazon_clothing"`, `"amazon_beauty"`, and `"kdd22"`. The default is set to `"kdd22"`.
- `MODEL`: Specifies the model to be used. The default model is `"gpt-4o-2024-08-06"`, but you can switch to other models such as `"claude-3-5-sonnet-20240620"` or `"qwen-plus-2024-09-19"`.
- `METHOD`: The method being used. In this case, it's set to `TAIRA`.
- `OPENAI_BASE_URL` and `OPENAI_API_KEY`: These fields are for configuring access to the OpenAI API. Make sure to provide your valid OpenAI API key. If you need to use a model other than openai, please set the corresponding base_url and api_key.
- `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`: These are for Google API configurations in Searcher Agent.

Ensure you have a valid OpenAI API key set up in the `OPENAI_API_KEY` field for the program to work properly.
<p id="Run"></p>  

## Running the Program

Once the configuration is set, you can run the main program with the following command:

```bash
python main.py
```
### Description:
- This command will execute the program based on the configurations defined in `system_config.yaml`.
- Ensure you have an active internet connection and valid API keys configured for proper operation.


[//]: # (## 💰 Simulation Cost)

[//]: # (🛎️ Note that all the experiments are powered by ChatGPT-3.5, and a complete simulation involving 1000 users would cost approximately $16. &#40;$0.016/User&#41;)


