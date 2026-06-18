Finviz Stock Screener presets (configurable)

How it works
- Presets are defined in Python as a fallback in `quantflow.data.finviz_client.PRESETS`.
- If `configs/finviz_presets.yml` exists, it overrides/extends the built-ins.
- CLI `make scan` iterates through all loaded presets.

YAML example (configs/finviz_presets.yml)
```
weekly_momo:
  - sh_avgvol_o300
  - sh_opt_option
  - ta_perf_1w10o
  - ta_sma50_pa
  - ta_sma200_pa
  - sh_price_o5
weekly_bear:
  - sh_avgvol_o300
  - sh_opt_option
  - ta_perf_1w-10u
  - ta_sma50_pb
  - sh_price_o5
reversal_bull:
  - sh_avgvol_o300
  - sh_opt_option
  - ta_rsi_os40
  - ta_sma50_pb
  - sh_price_o5
reversal_bear:
  - sh_avgvol_o300
  - sh_opt_option
  - ta_rsi_ob60
  - ta_sma50_pa
  - sh_price_o5
```

Notes
- Filter codes mirror Finviz UI selections. See comments in `finviz_client.py`.
- Add or remove presets in the YAML without changing code.
