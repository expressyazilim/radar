# --- Anomali maskesi ---
anom_mask = np.abs(Z_z) >= anomali_esik
anom_count = int(np.sum(anom_mask))

# --- Anomali merkezi (centroid) ---
anom_lat = None
anom_lon = None
if anom_count > 0:
    anom_lat = float(np.mean(Y[anom_mask]))
    anom_lon = float(np.mean(X[anom_mask]))

# --- 2D + 3D yan yana ---
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("🗺️ 2D Heatmap (VV dB)")

    heat_fig = go.Figure()

    heat_fig.add_trace(go.Heatmap(
        z=Z_db_clip,
        x=X[0, :],
        y=Y[:, 0],
        colorbar=dict(title="VV (dB)")
    ))

    # Kontur (anomali bölgeleri)
    # Z_norm üzerinden kontur daha stabil
    heat_fig.add_trace(go.Contour(
        z=anom_mask.astype(int),
        x=X[0, :],
        y=Y[:, 0],
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=1),
        line=dict(width=3),
        hoverinfo="skip",
        name="Anomali"
    ))

    # Merkez işareti
    if anom_lat is not None:
        heat_fig.add_trace(go.Scatter(
            x=[anom_lon],
            y=[anom_lat],
            mode="markers+text",
            text=["🎯"],
            textposition="top center",
            marker=dict(size=14),
            name="Merkez"
        ))

    heat_fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Boylam",
        yaxis_title="Enlem",
        title="2D Isı Haritası + Anomali Konturu"
    )

    st.plotly_chart(heat_fig, use_container_width=True)

    # Hızlı özet
    st.markdown("### 🧭 Anomali Özeti")
    if anom_count == 0:
        st.info("Bu eşikte belirgin anomali görünmüyor. Eşiği düşürmeyi deneyebilirsin.")
    else:
        st.success(f"Anomali piksel sayısı: **{anom_count}**")
        st.write(f"Merkez (tahmini): **{anom_lat:.6f}, {anom_lon:.6f}**")

with colB:
    st.subheader("🧊 3D Surface (VV dB)")

    surf_fig = go.Figure(data=[go.Surface(
        z=Z_db_clip,
        x=X,
        y=Y,
        colorscale="Viridis",
        hovertemplate=(
            "<b>Boylam</b>: %{x:.6f}<br>"
            "<b>Enlem</b>: %{y:.6f}<br>"
            "<b>VV (dB)</b>: %{z:.2f}<br><extra></extra>"
        )
    )])

    surf_fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.5),
            xaxis_title="Boylam",
            yaxis_title="Enlem",
            zaxis_title="VV (dB)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520,
        title="3D dB Yüzeyi"
    )

    st.plotly_chart(surf_fig, use_container_width=True)
