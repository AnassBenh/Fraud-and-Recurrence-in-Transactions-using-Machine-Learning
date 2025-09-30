import pandas as pd
import numpy as np
import cx_Oracle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import tkinter as tk
from tkinter import ttk

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, db_user, db_password, db_dsn):
        """Initialise le détecteur de fraude avec les paramètres de connexion à la base"""
        self.db_user = db_user
        self.db_password = db_password
        self.db_dsn = db_dsn
        self.conn = None
        self.model = None
        self.scaler = None
        
    def connect_db(self):
        """Établit la connexion à la base de données Oracle"""
        try:
            self.conn = cx_Oracle.connect(
                user=self.db_user,
                password=self.db_password,
                dsn=self.db_dsn
            )
            logger.info("Connexion à la base de données établie avec succès")
        except cx_Oracle.DatabaseError as e:
            logger.error(f"Erreur de connexion à la base de données: {e}")
            raise

    def fetch_transaction_data(self):
        """Récupère les données de transactions depuis la base de données"""
        query = """
        SELECT 
            t.TRANSACTION_ID, 
            t.CLIENT_ID, 
            c.NOM as client_nom,
            c.PRENOM as client_prenom,         
            t.COMMERCANT_ID, 
            m.NOM_ENTREPRISE as commercant_nom,
            t.DATE_TRANSACTION, 
            t.MONTANT, 
            t.STATUT,
            t.RECURRENTE,
            (SELECT COUNT(*) FROM transactions WHERE CLIENT_ID = t.CLIENT_ID) as NB_TRANSACTIONS_CLIENT,
            (SELECT AVG(MONTANT) FROM transactions WHERE CLIENT_ID = t.CLIENT_ID) as AVG_MONTANT_CLIENT,
            (SELECT COUNT(*) FROM transactions WHERE COMMERCANT_ID = t.COMMERCANT_ID) as NB_TRANSACTIONS_COMMERCANT,
            (SELECT AVG(MONTANT) FROM transactions WHERE COMMERCANT_ID = t.COMMERCANT_ID) as AVG_MONTANT_COMMERCANT
        FROM transactions t
        JOIN clients c ON t.CLIENT_ID = c.CLIENT_ID
        JOIN commercants m ON t.COMMERCANT_ID = m.COMMERCANT_ID
        """
        try:
            df = pd.read_sql(query, self.conn)
            df.columns = df.columns.str.lower()
            logger.info(f"{len(df)} transactions chargées depuis la base de données")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise

    def prepare_features(self, df):
        """Prépare les caractéristiques pour le modèle"""
        try:
            df['date_transaction'] = pd.to_datetime(df['date_transaction'])
            df['jour_semaine'] = df['date_transaction'].dt.dayofweek
            df['heure'] = df['date_transaction'].dt.hour
            df['mois'] = df['date_transaction'].dt.month

            df['diff_moyenne_client'] = df['montant'] - df['avg_montant_client']
            df['diff_moyenne_commercant'] = df['montant'] - df['avg_montant_commercant']

            df = df.sort_values(['client_id', 'date_transaction'])
            df['temps_entre_transactions'] = df.groupby('client_id')['date_transaction'].diff().dt.total_seconds() / 3600

            df['recurrente'] = df['recurrente'].map({'OUI': 1, 'NON': 0}).fillna(0)

            features = [
                'montant',
                'nb_transactions_client',
                'avg_montant_client',
                'nb_transactions_commercant',
                'avg_montant_commercant',
                'jour_semaine',
                'heure',
                'mois',
                'diff_moyenne_client',
                'diff_moyenne_commercant',
                'temps_entre_transactions',
                'recurrente'
            ]
            return df[features]
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des caractéristiques: {e}")
            raise

    def train_model(self, X, contamination=0.05):
        """Entraîne le modèle Isolation Forest"""
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=42,
                    verbose=1,
                    n_jobs=-1
                ))
            ])
            pipeline.fit(X)
            self.model = pipeline
            logger.info("Modèle entraîné avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            raise

    def predict_anomalies(self, df):
        """Prédit les anomalies et retourne le DataFrame enrichi"""
        try:
            X = self.prepare_features(df)
            X = X.fillna(0)
            df['anomaly_score'] = self.model.decision_function(X)
            df['is_fraud'] = self.model.predict(X)
            df['is_fraud'] = df['is_fraud'].map({1: 0, -1: 1})  # 1=frauduleuse, 0=normale
            fraud_count = df['is_fraud'].value_counts()
            logger.info(f"Transactions normales: {fraud_count.get(0, 0)}")
            logger.info(f"Transactions suspectes: {fraud_count.get(1, 0)}")
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction des anomalies: {e}")
            raise

    def prepare_database(self):
        """Prépare la base de données en ajoutant les colonnes nécessaires"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                BEGIN
                    EXECUTE IMMEDIATE 'ALTER TABLE transactions ADD (fraud_score NUMBER)';
                EXCEPTION
                    WHEN OTHERS THEN NULL;
                END;
            """)
            cursor.execute("""
                BEGIN
                    EXECUTE IMMEDIATE 'ALTER TABLE transactions ADD (is_fraud NUMBER(1))';
                EXCEPTION
                    WHEN OTHERS THEN NULL;
                END;
            """)
            self.conn.commit()
            logger.info("Base de données préparée avec succès")
        except cx_Oracle.DatabaseError as e:
            self.conn.rollback()
            logger.error(f"Erreur lors de la préparation de la base: {e}")
            raise

    def save_results(self, df):
        """Sauvegarde les résultats dans la base de données"""
        try:
            cursor = self.conn.cursor()
            data = [(row['anomaly_score'], row['is_fraud'], row['transaction_id']) 
                    for _, row in df.iterrows()]
            cursor.executemany("""
                UPDATE transactions
                SET fraud_score = :1,
                    is_fraud = :2
                WHERE transaction_id = :3
            """, data)
            self.conn.commit()
            logger.info(f"{len(data)} résultats sauvegardés dans la base de données")
        except cx_Oracle.DatabaseError as e:
            self.conn.rollback()
            logger.error(f"Erreur lors de la sauvegarde des résultats: {e}")
            raise

    def visualize_results(self, df):
        """Génère des visualisations des résultats"""
        try:
            plt.figure(figsize=(18, 12))
            # 1. Boxplot montants
            plt.subplot(2, 2, 1)
            sns.boxplot(x='is_fraud', y='montant', data=df)
            plt.title('Distribution des montants par statut de fraude')
            # 2. Histogramme des scores d'anomalie
            plt.subplot(2, 2, 2)
            sns.histplot(data=df, x='anomaly_score', hue='is_fraud', bins=50, kde=True)
            plt.title('Distribution des scores d\'anomalie')
            # 3. Transactions par jour de semaine
            plt.subplot(2, 2, 3)
            sns.countplot(data=df, x='jour_semaine', hue='is_fraud')
            plt.title('Transactions par jour de semaine')
            # 4. Montant vs Score d'anomalie
            plt.subplot(2, 2, 4)
            df['montant'] = pd.to_numeric(df['montant'], errors='coerce')
            df['anomaly_score'] = pd.to_numeric(df['anomaly_score'], errors='coerce')
            plt.scatter(
                df[df['is_fraud'] == 0]['montant'],
                df[df['is_fraud'] == 0]['anomaly_score'],
                c='blue', label='Normale', alpha=0.5
            )
            plt.scatter(
                df[df['is_fraud'] == 1]['montant'],
                df[df['is_fraud'] == 1]['anomaly_score'],
                c='red', label='Suspecte', alpha=0.7
            )
            plt.xlabel('Montant')
            plt.ylabel('Score d\'anomalie')
            plt.title('Montant vs Score d\'anomalie')
            plt.legend()
            plt.tight_layout()
            plt.savefig('fraud_analysis.png')
            logger.info("Visualisations sauvegardées dans fraud_analysis.png")
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {e}")

    def run(self):
        """Exécute en complet la détection de fraude"""
        try:
            self.connect_db()
            df = self.fetch_transaction_data()
            X = self.prepare_features(df)
            X = X.fillna(0)
            self.train_model(X)
            df = self.predict_anomalies(df)
            self.prepare_database()
            self.save_results(df)
            self.visualize_results(df)
            return df
        except Exception as e:
            logger.error(f"Erreur dans l'exécution du pipeline: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Connexion à la base de données fermée")

if __name__ == "__main__":
    config = {
        'db_user': 'anass',
        'db_password': '1234',
        'db_dsn': 'localhost/XEPDB1'
    }
    detector = FraudDetector(**config)
    results_df = detector.run()

    # Ajoute les features nécessaires pour l'explication
    features = detector.prepare_features(results_df)
    results_df['diff_moyenne_client'] = features['diff_moyenne_client']
    results_df['diff_moyenne_commercant'] = features['diff_moyenne_commercant']
    results_df['temps_entre_transactions'] = features['temps_entre_transactions']
    results_df['recurrente'] = features['recurrente']
    results_df['avg_montant_client'] = features['avg_montant_client']
    results_df['avg_montant_commercant'] = features['avg_montant_commercant']

    def get_explication(row):
        explications = []
        if abs(row['diff_moyenne_client']) > abs(row['avg_montant_client']) * 0.5:
            explications.append("Montant très différent de la moyenne client")
        if abs(row['diff_moyenne_commercant']) > abs(row['avg_montant_commercant']) * 0.5:
            explications.append("Montant très différent de la moyenne commerçant")
        if row['temps_entre_transactions'] is not None and row['temps_entre_transactions'] < 1:
            explications.append("Transactions très rapprochées")
        if row['recurrente'] == 0:
            explications.append("Transaction non récurrente")
        return "; ".join(explications) if explications else "Profil global atypique"

    results_df['explication'] = results_df.apply(get_explication, axis=1)

    suspects = results_df[results_df['is_fraud'] == 1][
        ['transaction_id', 'client_id', 'client_prenom', 'client_nom', 'montant', 'anomaly_score', 'explication']
    ].sort_values('anomaly_score', ascending=False)

    # Export CSV 
    suspects.to_csv("transactions_suspectes.csv", index=False)
    logger.info("Analyse de fraude terminée, résultats exportés dans transactions_suspectes.csv")
