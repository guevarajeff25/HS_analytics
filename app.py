with right:
    st.markdown("### Pitching Leaders")

    if pit_m is None or pit_m.empty:
        st.info("Load pitching to see leaders.")
    else:
        pit_show = pit_m.copy()

        # ---- Filter pitchers with innings > 0 using outs-aware IP_TRUE if present ----
        if "IP_TRUE" in pit_show.columns:
            pit_show["_IP_TRUE_NUM"] = pd.to_numeric(pit_show["IP_TRUE"], errors="coerce").fillna(0)
            pit_show = pit_show[pit_show["_IP_TRUE_NUM"] > 0].copy()
        else:
            # fallback if somehow IP_TRUE isn't present
            pit_show["_IP_NUM"] = pd.to_numeric(pit_show.get("IP", 0), errors="coerce").fillna(0)
            pit_show = pit_show[pit_show["_IP_NUM"] > 0].copy()

        # ---- HARD DROP: never allow helper/legacy cols in tables ----
        pit_show = pit_show.drop(
            columns=[c for c in ["IP_DISPLAY", "IP_TRUE", "_IP_TRUE_NUM", "_IP_NUM"] if c in pit_show.columns],
            errors="ignore",
        )

        if pit_show.empty:
            st.info("No pitchers with IP > 0 found in this dataset.")
        else:
            tabs = st.tabs(["WHIP (low)", "K/BB", "K/BF", "BB/INN (low)", "ERA (low)"])
            dec = {"WHIP": 2, "K/BB": 2, "K/BF": 3, "BB/INN": 2, "ERA": 2}

            def _format_pit(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                for col, d in dec.items():
                    if col in out.columns:
                        out[col] = pd.to_numeric(out[col], errors="coerce").apply(lambda v: fmt_no0(v, d))
                # IP should already be baseball notation string from utils
                if "IP" in out.columns:
                    out["IP"] = out["IP"].astype(str)
                return out

            with tabs[0]:
                df = top_table(pit_show, "WHIP", n=12, ascending=True)
                st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)

            with tabs[1]:
                df = top_table(pit_show, "K/BB", n=12, ascending=False)
                st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)

            with tabs[2]:
                df = top_table(pit_show, "K/BF", n=12, ascending=False)
                st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)

            with tabs[3]:
                df = top_table(pit_show, "BB/INN", n=12, ascending=True)
                st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)

            with tabs[4]:
                df = top_table(pit_show, "ERA", n=12, ascending=True)
                st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)
