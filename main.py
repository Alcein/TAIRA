from datetime import datetime

import pandas as pd
import logging
import os

import yaml

from manager import Manager
from executor_agents.item_retrieval_agent import ItemRetrievalAgent
from executor_agents.searcher import SearcherAgent
from executor_agents.interact_agent import InteractorAgent
from executor_agents.task_interpreter import InterpreterAgent
from utils.memory import Memory



pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.unicode.east_asian_width', True)


# Ensure the logs directory exists


def setup_logger(log_file):
    # Create a custom logger
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    f_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)

    return logger


def evaluate_template(domain, classification, template):
    rules = {
        "amazon_clothing": {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  # domain1情况下数据1与数据2的配对规则
        "amazon_beauty": {-1: 0, 0: 8, 1: 9, 2: 10, 3: 4, 4: 5},  # domain2情况下数据1与数据2的配对规则
        "amazon_music": {-1: 0, 0: 11, 1: 12, 2: 13, 3: 4, 4: 5}  # domain3情况下数据1与数据2的配对规则
    }
    if domain in rules:
        return rules[domain].get(classification) == template
    else:
        return False


def main():
    # Load the CSV file
    memory = Memory()
    item_agent = ItemRetrievalAgent(memory)
    searcher_agent = SearcherAgent(memory)
    interactor_agent = InteractorAgent(memory)
    interpreter = InterpreterAgent(memory)

    with open('system_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    domain = config['DOMAIN']
    method = config['METHOD']
    query_number = config['QUERY_NUMBER']
    domain_path = "data/" + domain
    df = pd.read_csv(domain_path + '/query_data1.csv', encoding='ISO-8859-1').head(query_number)
    df = df[df['classification'] == 1]
    start_num = 0
    df = df.iloc[start_num:]
    # df = df.iloc[[1,3,5,7,9,12,13,15,18,19,20,21,25,26,28,32,35,36,37,38,40,44,45,46,47,48]]

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H_%M_%S")
    log_dir = f'{domain_path}/logs/{method}-{formatted_time}'
    os.makedirs(log_dir, exist_ok=True)

    # Output CSV file path
    new_csv_path = log_dir + f'/result{method}-{formatted_time}.csv'

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        log_file = f'{log_dir}/log_{index + 1}.log'
        logger = setup_logger(log_file)
        print("正在处理query", index + 1)
        user_input = row['new_query']
        target_count = row['target_count']
        targets = row['targets']
        user_preference = row['preferences']

        if domain == "amazon_clothing" or domain == "amazon_music":
            target_product = f"{row['title']} | {row['category']}"
        elif domain == "amazon_beauty":
            target_product = f"{row['title']} | {row['description']} | {row['category']}"
        else:
            target_product = "no target"
        attempt = 0
        try:
            if method in ['TAIRA']:
                manager = Manager(memory, user_input, target_product, targets, target_count, user_preference, config,
                                  logger=logger)
            else:
                print('method error')
                return

            manager.register_agent(item_agent)
            manager.register_agent(searcher_agent)
            manager.register_agent(interactor_agent)
            manager.register_agent(interpreter)

            result = manager.delegate_task()
            # print(result)
            hit_rate, mrr, ndcg, fail_flag, template = result
            if method == 'TAIRA':
                template_suc = evaluate_template(domain, int(row['classification']), template)
            else:
                template_suc = False

        except Exception as e:
            error_message = f"Error processing query {index + 1}: {str(e)}"
            print(error_message)
            logger.error(error_message)
            memory.remove_data()
            logger.handlers.clear()
            continue

        # Append the result of the current query to the DataFrame
        row['hit_rate'] = hit_rate
        row['mrr'] = mrr
        row['ndcgs'] = ndcg
        row['fail'] = 1 if fail_flag else 0
        row['template'] = template
        row['template_success'] = template_suc

        # Write the current row to the CSV (append mode)
        row_df = pd.DataFrame([row])
        if not os.path.exists(new_csv_path):
            row_df.to_csv(new_csv_path, mode='w', header=True, index=False)
        else:
            row_df.to_csv(new_csv_path, mode='a', header=False, index=False)

        # Reset memory for the next iteration
        memory.remove_data()
        logger.handlers.clear()
    complete_df = pd.read_csv(new_csv_path, encoding='ISO-8859-1')
    mean_hit_rate = complete_df['hit_rate'].mean()
    mean_mrr = complete_df['mrr'].mean()
    mean_ndcgs = complete_df['ndcgs'].mean()
    mean_fail = 1 - complete_df['fail'].mean()

    # 创建一个包含均值的新行
    mean_row = pd.DataFrame({
        'hit_rate': [mean_hit_rate],
        'mrr': [mean_mrr],
        'ndcgs': [mean_ndcgs],
        'fail': [mean_fail]
    })

    mean_row.to_csv(new_csv_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    main()
