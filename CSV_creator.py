import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# On définit le nombre total de transactions et le pourcentage de fraudes
n_transactions = 10000  # Total de transactions
fraud_ratio = 0.02  # 2% de transactions frauduleuses


# Fonction pour créer des transactions
def generate_transactions(n, fraud_ratio):
    data = []
    fraud_count = int(n * fraud_ratio)  # Calcul du nombre de transactions frauduleuses

    for i in range(n):
        # Infos de base de la transaction
        transaction_id = f"T{i + 1:07d}"  # Identifiant unique pour chaque transaction

        # Génération d'une date et d'une heure aléatoires entre 2015 et 2024
        random_year = random.randint(2015, 2024)
        random_date = datetime(random_year, random.randint(1, 12), random.randint(1, 28))
        random_time = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        timestamp = random_date + random_time

        # Montant variable pour refléter des habitudes de dépenses
        amount = round(np.random.exponential(50) * random.uniform(0.5, 10), 2)
        merchant_id = f"M{random.randint(1, 500):04d}"
        customer_id = f"C{random.randint(1, 1000):04d}"

        # Type de transaction et influence sur le montant moyen
        transaction_type = random.choice(["payment", "withdrawal", "deposit", "transfer"])
        if transaction_type == "withdrawal":
            amount = round(amount * 0.75, 2)
        elif transaction_type == "deposit":
            amount = round(amount * 1.2, 2)

        # Pays et catégorie de marchand avec plus de détails
        country = random.choice(["FR", "US", "DE", "UK", "ES", "IT", "JP", "CN"])
        device_id = f"D{random.randint(1, 3000):04d}"
        ip_address = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
        merchant_category = random.choice(["retail", "restaurant", "technology", "travel", "entertainment"])

        # Heure et jour de la transaction influencés par la catégorie de marchand
        hour_of_day = timestamp.hour
        if merchant_category == "restaurant":
            hour_of_day = random.choice(range(11, 24))  # Transactions au restaurant souvent le soir
        elif merchant_category == "retail":
            hour_of_day = random.choice(range(9, 21))  # Heures d'ouverture pour les commerces de détail

        timestamp = timestamp.replace(hour=hour_of_day)

        # Si la transaction est frauduleuse
        is_fraud = 1 if i < fraud_count else 0
        if is_fraud:
            amount *= random.uniform(1.5, 3)
            transaction_type = random.choice(["payment", "withdrawal"])
            country = random.choice(["US", "CN", "RU"])
            device_id = f"D{random.randint(3000, 5000):04d}"
            ip_address = f"10.0.{random.randint(0, 255)}.{random.randint(0, 255)}"

        # On ajoute toutes ces infos à notre liste
        data.append([
            transaction_id, timestamp, amount, merchant_id,
            customer_id, transaction_type, country, device_id,
            ip_address, merchant_category, hour_of_day, is_fraud
        ])

    # On crée un DataFrame avec les colonnes appropriées
    return pd.DataFrame(data, columns=[
        "transaction_id", "timestamp", "amount", "merchant_id",
        "customer_id", "transaction_type", "country", "device_id",
        "ip_address", "merchant_category", "hour_of_day", "is_fraud"
    ])


# On génère le dataset
df = generate_transactions(n_transactions, fraud_ratio)

# Sauvegarde en fichier CSV
df.to_csv("transactions.csv", index=False)
print("Dataset généré et sauvegardé sous le nom 'transactions.csv'") 