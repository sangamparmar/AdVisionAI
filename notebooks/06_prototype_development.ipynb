{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-07-31 18:12:43.491 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\sanga\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "st.title(\"AI-Driven Media Investment Plan\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload Ad Spend Data\", type=\"csv\")\n",
    "if uploaded_file is not None:\n",
    "    ad_spend_data = pd.read_csv(uploaded_file)\n",
    "    st.write(ad_spend_data.head())\n",
    "    \n",
    "    # Prepare data for model\n",
    "    X = ad_spend_data[['impressions', 'clicks']]\n",
    "    y = ad_spend_data['conversions']\n",
    "    \n",
    "    # Train model\n",
    "    model = LinearRegression()\n",
    "    model.fit(X, y)\n",
    "    \n",
    "    total_budget = st.number_input(\"Enter Total Budget\", value=100000)\n",
    "    channels = ad_spend_data['channel'].unique()\n",
    "    \n",
    "    channel_budgets = {}\n",
    "    for channel in channels:\n",
    "        channel_data = ad_spend_data[ad_spend_data['channel'] == channel]\n",
    "        predicted_conversion = model.predict(channel_data[['impressions', 'clicks']].mean().values.reshape(1, -1))[0]\n",
    "        allocated_budget = max(0.1 * total_budget, predicted_conversion)\n",
    "        channel_budgets[channel] = allocated_budget\n",
    "    \n",
    "    st.write(\"Reallocated Budgets:\")\n",
    "    st.write(channel_budgets)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
