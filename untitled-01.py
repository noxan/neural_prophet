# %% [markdown]
# # Holiday regularization #133

# %%
import pandas as pd

from neuralprophet import NeuralProphet, set_log_level

set_log_level("ERROR")

# %%
df = pd.read_csv("tests/test-data/air_passengers.csv")

# %%
m = NeuralProphet()

# add the country specific holidays
# m = m.add_country_holidays("US")
# m = m.add_country_holidays("US", regularization=0.5)
# m = m.add_country_holidays("US", regularization=1)
# m = m.add_country_holidays("US", regularization=1e10)
# m = m.add_country_holidays("Indonesia")

m = m.add_events("birthday")  # , regularization=1)

events_df = pd.DataFrame(
    {
        "event": "birthday",
        "ds": pd.to_datetime(["1960-07-01"]),
    }
)
history_df = m.create_df_with_events(df, events_df)

# fit the model
metrics = m.fit(history_df, freq="MS")

future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)

forecast = m.predict(df=future)

# %%

m.model.events_dims

# %%

m.model.get_event_weights("birthday")

# %%
