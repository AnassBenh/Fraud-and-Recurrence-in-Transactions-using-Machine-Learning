import cx_Oracle
import pandas as pd
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

conn = cx_Oracle.connect("anass", "1234", "localhost/XEPDB1")
cursor = conn.cursor()
cursor.execute("""
    SELECT TRANSACTION_ID, CLIENT_ID, COMMERCANT_ID, MONTANT, DATE_TRANSACTION, MOYEN_PAIEMENT, STATUT, RECURRENTE
    FROM transactions
""")
rows = cursor.fetchall()

# Mise en forme en DataFrame avec les noms de colonnes en majuscules
df = pd.DataFrame(rows, columns=[
    'TRANSACTION_ID', 'CLIENT_ID', 'COMMERCANT_ID', 'MONTANT',
    'DATE_TRANSACTION', 'MOYEN_PAIEMENT', 'STATUT', 'RECURRENTE'
])
df['DATE_TRANSACTION'] = pd.to_datetime(df['DATE_TRANSACTION'])
df['RECURRENTE'] = df['RECURRENTE'].fillna('NON')

# Détection des transactions récurrentes
for (client, commercant), group in df.groupby(['CLIENT_ID', 'COMMERCANT_ID']):
    group = group.sort_values('DATE_TRANSACTION')
    for i in range(1, len(group)):
        diff = (group.iloc[i]['DATE_TRANSACTION'] - group.iloc[i - 1]['DATE_TRANSACTION']).days
        montant_diff = abs(group.iloc[i]['MONTANT'] - group.iloc[i - 1]['MONTANT'])
        if 28 <= diff <= 32 and montant_diff < 1:
            df.loc[group.index[i], 'RECURRENTE'] = 'OUI'
            df.loc[group.index[i - 1], 'RECURRENTE'] = 'OUI'

# Mise à jour de la base de données
for _, row in df.iterrows():
    cursor.execute("""
        UPDATE transactions
        SET RECURRENTE = :1
        WHERE TRANSACTION_ID = :2
    """, (row['RECURRENTE'], row['TRANSACTION_ID']))

print(df[df['RECURRENTE'] == 'OUI'])
conn.commit()
logger.info("Mise à jour terminée.")