import pandas as pd
import numpy as np
from tqdm import tqdm 

if __name__ == "__main__":

    user_path = r"C:\Users\srija\Desktop\FYP\Data\userListFromExtraction.csv"
    email_path = r"C:\Users\srija\Desktop\r4.2\r4.2\email.csv"
    
    users = pd.read_csv(user_path)
    emails = pd.read_csv(email_path)
    
    user_dict = {id: i for (i, id) in enumerate(users.uid)}
    
    n = len(users)
    adj_matrix = np.zeros((n, n))
    
    for i in tqdm(range(len(emails)), desc="Processing Emails"):
        row = emails.iloc[i]
        sender = row['from']
        
        if 'dtaa.com' not in sender:
            continue
        
        receivers = row['to'].split(';')
        
        if type(row['cc']) == str:
            receivers = receivers + row['cc'].split(";")
        if type(row['bcc']) == str:
            receivers = receivers + row['bcc'].split(";")   
        
        sender = users[users['email']==sender]['uid'].iloc[0]
        receivers = [users[users['email']==email]['uid'].iloc[0] for email in receivers if 'dtaa.com' in email]
            
        for receiver in receivers:
            adj_matrix[user_dict[sender], user_dict[receiver]] = 1
            adj_matrix[user_dict[receiver], user_dict[sender]] = 1
    
    for i in tqdm(range(len(users)), desc="Processing Supervisors"):
        row = users.iloc[i]
        user = row['uid']
        if type(row['sup']) == str:
            sup = users[users['uid'] == row['sup']]['uid'].iloc[0]
            adj_matrix[user_dict[user], user_dict[sup]] = 1
            adj_matrix[user_dict[user], user_dict[sup]] = 1
        
    adj_matrix_df = pd.DataFrame(adj_matrix, columns=users['uid'], index=users['uid'])
    adj_matrix_path = r"Knowledge_adjacency_matrix.csv"
    adj_matrix_df.to_csv(adj_matrix_path, index=True)