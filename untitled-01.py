# %% [markdown]
# # Holiday regularization #133

# %%
import pandas as pd

from neuralprophet import NeuralProphet, set_log_level

set_log_level("ERROR")

# %%
import pandas as pd


def generate_event_dataset(events=["2022-01-01", "2022-01-10", "2022-01-31"], periods=31, y_default=1, y_event=1000):
    """Generate dataset with regular y value and special y value for events."""
    events.sort()

    dates = pd.date_range(events[0], periods=periods, freq="D")
    df = pd.DataFrame({"ds": dates, "y": y_default}, index=dates)

    for event in events:
        df.loc[event, "y"] = y_event

    return df, events


df, events = generate_event_dataset()

# %%
m = NeuralProphet()

# add the country specific holidays
# m = m.add_country_holidays("US")
# m = m.add_country_holidays("US", regularization=0.5)
# m = m.add_country_holidays("US", regularization=1)
# m = m.add_country_holidays("US", regularization=1e10)
# m = m.add_country_holidays("Indonesia")

# m = m.add_events("birthday")
m = m.add_events(["event_%i" % index for index, event in enumerate(events)], regularization=1)


events_df = pd.concat(
    [
        pd.DataFrame(
            {
                "event": "event_%i" % index,
                "ds": pd.to_datetime([event]),
            }
        )
        for index, event in enumerate(events)
    ]
)

history_df = m.create_df_with_events(df, events_df)

#%%

# fit the model
metrics = m.fit(history_df, freq="MS")

future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)

forecast = m.predict(df=future)

# %%

m.model.get_event_weights("event_0")

# %%

history_df.iloc[-20:]
