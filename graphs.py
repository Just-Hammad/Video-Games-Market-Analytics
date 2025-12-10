# ----------------------------
# HEATMAP: Release Month × Price Band → Average Playtime Forever
# ----------------------------
indie = df[df["indie"] == True].copy()
indie['ReleaseMonth'] = indie['Release date'].dt.month
playtime_heatmap = indie.groupby(['ReleaseMonth', 'price_bin'])['Average playtime forever'].mean().unstack()

plt.figure(figsize=(14, 6))
sns.heatmap(playtime_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
plt.title("Release Month × Price Band → Average Playtime Forever (Indie Games)")
plt.xlabel("Price Band")
plt.ylabel("Release Month")
plt.yticks(np.arange(0.5, 12.5), labels=range(1, 13), rotation=0)  # months 1-12
plt.tight_layout()
plt.show()



# =====================================================
# GRAPH 2: PRICE BANDS VS SUCCESS FOR INDIE GAMES
# =====================================================
print("\n[2/10] PRICE BANDS VS SUCCESS FOR INDIES")
print("="*80)

# ------------------------
# Filtering indie games
# ------------------------
indie_games = df[df['indie'] == True].copy()

# ------------------------
# Create price bands
# ------------------------
price_bins = [-1, 0, 5, 10, 20, 30, np.inf]
price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']

indie_games['price_band'] = pd.cut(
    indie_games['Price'],
    bins=price_bins,
    labels=price_labels,
    include_lowest=True
)

# Keep correct order in plotting
price_order = price_labels

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(12, 6))

primary_color = "#FF6B6B"      # consistent base theme
highlight_color = "#4ECDC4"    # not needed here but kept for consistency

sns.boxplot(
    data=indie_games,
    x='price_band',
    y='Steam Score',
    order=price_order,
    color=primary_color,
    fliersize=3,
    linewidth=1.3
)

# Styling
plt.xlabel("Price Band", weight='bold')
plt.ylabel("Steam Score", weight='bold')
plt.title("Success Distribution of Indie Games Across Price Bands", weight='bold')
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Summary insight
summary = (
    indie_games.groupby('price_band')['Steam Score']
    .median()
    .reset_index()
    .sort_values('Steam Score', ascending=False)
)

print("\n✓ Median Steam Score per price band:")
for _, row in summary.iterrows():
    print(f"  {row['price_band']}: {row['Steam Score']:.2f}")




import matplotlib.pyplot as plt

# ------------------------
# Sort by avg_success for plotting
# ------------------------
genre_stats = genre_stats.sort_values('avg_success', ascending=False)
# ------------------------
# GENRE ANALYSIS with EstimatedProfit
# ------------------------
genre_stats = (
    indie_games[indie_games['primary_genre'].notna()]
    .groupby('primary_genre')
    .agg(
        avg_success=('Steam Score', lambda x: (x >= 0.7).mean() * 100),
        count=('Name', 'count'),
        EstimatedProfit=('EstimatedProfit', 'mean')  # <-- add this
    )
    .reset_index()
)

# Keep only genres with at least 100 indie games
genre_stats = genre_stats[genre_stats['count'] >= 100]

# Sort by success rate for plotting
genre_stats = genre_stats.sort_values('avg_success', ascending=False)


# ------------------------
# PLOTTING: Vertical bars + profit line
# ------------------------
fig, ax1 = plt.subplots(figsize=(14, 7))

x = range(len(genre_stats))

# Bar: Average Success Rate
bars = ax1.bar(
    x, 
    genre_stats['avg_success'], 
    color="#FF6B6B", 
    edgecolor='black', 
    linewidth=1.2, 
    alpha=0.85
)
ax1.set_xticks(x)
ax1.set_xticklabels(genre_stats['primary_genre'], rotation=45, ha='right')
ax1.set_ylabel("Average Success Rate (%)", weight='bold')
ax1.set_title("Best Indie Genres by Player Rating & Estimated Profit", weight='bold')
ax1.grid(axis='y', alpha=0.3)

# # Annotate bars with success %
# for bar, val in zip(bars, genre_stats['avg_success']):
#     ax1.text(
#         bar.get_x() + bar.get_width()/2, 
#         val + 1, 
#         f"{val:.1f}%", 
#         ha='center', 
#         va='bottom',
#         fontweight='bold'
#     )

# Line: Mean Estimated Profit
ax2 = ax1.twinx()
ax2.plot(
    x, 
    genre_stats['EstimatedProfit'], 
    color="#FFD93D", 
    marker='o', 
    liw3newidth=2, 
    label='Mean Estimated Profit'
)
ax2.set_ylabel("Mean Estimated Profit ($)", weight='bold')

# Annotate profit line
for i, val in enumerate(genre_stats['EstimatedProfit']):
    ax2.text(
        i, 
        val + val*0.02, 
        f"${val:,.0f}", 
        ha='center', 
        va='bottom', 
        fontweight='bold', 
        color='#FFD93D'
    )

plt.tight_layout()
plt.show()
