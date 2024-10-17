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




def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    f_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    f_handler.setLevel(logging.DEBUG)

    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(f_handler)

    return logger

def evaluate_template(domain, classification, template):
    rules = {
        "amazon_clothing": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "amazon_beauty": {0: 8, 1: 9, 2: 10, 3: 4, 4: 5},
        "kdd22": {1: 6, 2: 7}
    }
    if domain in rules:
        return rules[domain].get(classification) == template
    else:
        return False


def main():
    memory = Memory()
    item_agent = ItemRetrievalAgent(memory)
    searcher_agent = SearcherAgent(memory)
    interactor_agent = InteractorAgent(memory)
    interpreter = InterpreterAgent(memory)
    points = []
    mrrs = []
    ndcgs = []
    fail = []
    template_suc = []

    with open('system_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    domain = config['DOMAIN']
    method = config['METHOD']
    query_number = config['QUERY_NUMBER']
    domain_path = "data/" + domain
    df = pd.read_parquet(domain_path + '/query_data1.parquet', encoding='ISO-8859-1').head(query_number)
    start_num = 0
    df = df.iloc[start_num:]

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H_%M_%S")
    log_dir = f'{domain_path}/logs/{method}-{formatted_time}'
    os.makedirs(log_dir, exist_ok=True)

    new_csv_path = log_dir + f'/result{method}-{formatted_time}.csv'

    for index, row in df.iterrows():
        log_file = f'{log_dir}/log_{index + 1}.log'
        logger = setup_logger(log_file)
        try:
            print("正在处理query", index + 1)
            user_input = row['new_query']
            complements = ''
            target_count = row['target_count']
            targets = row['targets']

            if domain == "amazon_clothing":
                target_product = f"{row['title']} | {row['categories']}"
            elif domain == "amazon_beauty":
                target_product = f"{row['title']} | {row['description']} | {row['categories']}"
            elif domain == "kdd22":
                target_product = row['product_info']
                complements = row['complements']
                if pd.isna(targets):
                    targets = ""
            else:
                target_product = "no target"
            manager = Manager(memory, user_input, target_product, complements, targets, target_count, logger=logger)

            manager.register_agent(item_agent)
            manager.register_agent(searcher_agent)
            manager.register_agent(interactor_agent)
            manager.register_agent(interpreter)

            result = manager.delegate_task()
            hit_rate, mrr, ndcg, fail_flag, template = result

            points.append(hit_rate)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            fail.append(1 if fail_flag else 0)
            if method == 'TAIRA':
                template_suc.append(evaluate_template(domain, int(row['classification']), template))
            else:
                template_suc.append(False)
        except Exception as e:
            error_message = f"Error processing query {index + 1}: {str(e)}"
            print(error_message)
            logger.error(error_message)
            hit_rate = -1
            mrr = -1
            ndcg = -1
            fail_flag = 'error'

            points.append(hit_rate)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            fail.append(fail_flag)
            template_suc.append(False)
            logger.handlers.clear()

        row['hit_rate'] = hit_rate
        row['mrr'] = mrr
        row['ndcgs'] = ndcg
        row['fail'] = 1 if fail_flag else 0
        row['template_success'] = template_suc[-1]

        row_df = pd.DataFrame([row])
        if not os.path.exists(new_csv_path):
            row_df.to_csv(new_csv_path, mode='w', header=True, index=False)
        else:
            row_df.to_csv(new_csv_path, mode='a', header=False, index=False)

        memory.remove_data()
        logger.handlers.clear()



if __name__ == "__main__":
    main()

