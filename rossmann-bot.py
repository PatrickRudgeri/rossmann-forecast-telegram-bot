import pandas as pd
from flask import Flask, Response, request
import requests
import json
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

# ---- CONSTANTS
TOKEN = os.getenv("TOKEN")
ENDPOINT_PROD = os.getenv("ENDPOINT_PROD")
ENDPOINT_DEV = os.getenv("ENDPOINT_DEV")
ENDPOINT_MODEL = os.getenv("ENDPOINT_MODEL")
BASE_URL = os.getenv("BASE_URL") + TOKEN

# ----

# set webhook and cleaning previous messages
# https://api.telegram.org/bot1508204454:AAH54BEN9QZTZz6NJDzT8Ti4Mr6sZT6W9x4/setWebhook?url=https://patrick-2764276b.localhost.run&drop_pending_updates=true

requests.get(f"{BASE_URL}/setWebhook?url={ENDPOINT_PROD}&drop_pending_updates=true")


# --------------

def df_test_stores(df:pd.DataFrame, stores) -> pd.DataFrame:
	""" choose store(s) for prediction """

	if isinstance(stores, list):
		df1 = df.query("Store in @stores")
	elif isinstance(stores, int):
		df1 = df.query("Store == @stores")
	else:
		print("stores must to be a list or int")
		return None

	df1 = df1[(df1["Open"] != 0) & (~df1["Open"].isnull())]
	df1.drop("Id", axis=1, inplace=True)
	return df1


def load_dataset(stores):
	# loading test dataset
	df_test_raw = pd.read_csv("data/test.csv")
	df_store_raw = pd.read_csv("data/store.csv")

	# merge test + store
	df_test = pd.merge(df_test_raw, df_store_raw, on="Store", how="left")

	# orientação == records (cada registro == um dict)
	data_stores = df_test_stores(df_test, stores)

	stores_data = None if len(data_stores) == 0 else json.dumps(data_stores.to_dict(orient="records"))

	return stores_data

def predict(stores_data):
	# API call
	url = f"{ENDPOINT_MODEL}/rossmann/predict"
	header = {"Content-type": "application/json"}
	# data = all_data
	data = stores_data

	r = requests.post(url, data=data, headers=header)
	print(f"Rossmann API Status code {r.status_code}")

	return pd.DataFrame(r.json())

def send_message(chat_id, text, reply_to_msg_id=None):
	""" Send a text message to chat_id """
	url = BASE_URL + f"/sendMessage?chat_id={chat_id}"

	data = {"text": text}

	if reply_to_msg_id is not None:
		data["reply_to_message_id"] = reply_to_msg_id

	r = requests.post(url, json=data)

	print("Status send message:", r.status_code)

def parse_message(message):
	chat_id = message["message"]["chat"]["id"]
	text = message["message"]["text"]
	text = text.replace("/", "")

	try:
		store_id = int(text)
	except ValueError:
		store_id = None


	return chat_id, store_id


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		message = request.get_json()

		chat_id, store_id = parse_message(message)

		if store_id is None:
			send_message(chat_id, "Store ID is wrong.")
			return Response("Ok", status=200)

		# Loading data
		data = load_dataset(store_id)
		if data is None:
			send_message(chat_id, "Store not available.")
			return Response("Ok", status=200)

		send_message(chat_id, f"Consultando previsões da loja {store_id} ...")

		# prediction
		d1 = predict(data)

		# calculation

		# # Vendas Totais das proximas 6 semanas
		sum_by_store = d1[["Store", "prediction"]].groupby("Store").sum().reset_index()

		# send message
		msg = f"A loja {store_id} venderá €{sum_by_store['prediction'].values[0]:,.2f} nas próximas 6 semanas."
		send_message(chat_id, msg)
		return Response("Ok", status=200)

	else:
		return "<h1>Rossmann Sales Predictions BOT for Telegram</h1>"



if __name__ == "__main__":
	port = os.environ.get('PORT', 5000)
	app.run(host="0.0.0.0", port=port)
