"use client";

import React, { useEffect, useState } from "react";
import { HeroParallax } from "@/components/ui/hero-parallax";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { TracingBeam } from "@/components/ui/tracing-beam";
import { HoverEffect } from "@/components/ui/hover-effect";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { BackgroundBeams } from "@/components/ui/background-beams";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    AreaChart, Area, BarChart, Bar, ScatterChart, Scatter, Cell, ReferenceLine
} from "recharts";
import Plot from "react-plotly.js";
import { Gamepad2, DollarSign, Users, Trophy } from "lucide-react";

// --- TYPES ---
interface KPIData {
    total_games: number;
    indie_share: number;
    median_indie_profit: number;
    luxury_share: number;
}
interface PriceTrend {
    year: number;
    avg_price: number;
    median_price: number;
}
interface ReleaseStrategy {
    Strategy: string;
    Total_Sales: number;
}
interface OwnershipTier {
    year: number;
    Budget: number;
    Standard: number;
    Premium: number;
    Luxury: number;
}
interface GenreSuccess {
    genre: string;
    success_rate: number;
    mean_profit: number;
    count: number;
}
interface IndieMatrix {
    genre: string;
    competition: number;
    success_rate: number;
    market_type: string;
    count: number;
}
interface PlatformData {
    platform: string;
    median_revenue: number;
    strategy: string;
}
interface SeasonalityData {
    quarter: string;
    traffic: number;
    competition: number;
    recommendation: string;
}
interface StrategyScatter {
    Name: string;
    Release_Span: number;
    Total_Sales: number;
    Platform_Count: number;
}
// NEW Option B types
interface PriceBand {
    band: string;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    count: number;
}
interface GenreLollipop {
    genre: string;
    success_rate: number;
    median_profit: number;
    count: number;
}
interface FilteredKPI {
    filter: string;
    total: number;
    avg_price: number;
    median_profit: number;
    avg_score: number;
}
interface HeatmapCell {
    month: string;
    price_band: string;
    avg_playtime: number;
}
// NEW: Zain Features
interface LaunchEcosystem {
    launch_type: string;
    count: number;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    mean: number;
}
interface ExecutionMultiplier {
    feature: string;
    traffic_lift: number;
    category: string;
}
interface ProMatrix {
    genre: string;
    safety: number;
    potential: number;
    market_type: string;
}
interface GenreAnalysis {
    Genre: string;
    Total_Traffic: number;
    Median_Traffic: number;
    Mean_Traffic: number;
    Game_Count: number;
    Inequality_Ratio: number;
}

export default function Dashboard() {
    const [kpi, setKpi] = useState<KPIData | null>(null);
    const [priceTrends, setPriceTrends] = useState<PriceTrend[]>([]);
    const [ownershipData, setOwnershipData] = useState<OwnershipTier[]>([]);
    const [strategyData, setStrategyData] = useState<ReleaseStrategy[]>([]);
    const [strategyScatter, setStrategyScatter] = useState<StrategyScatter[]>([]);
    const [genreSuccess, setGenreSuccess] = useState<GenreSuccess[]>([]);
    const [indieMatrix, setIndieMatrix] = useState<IndieMatrix[]>([]);
    const [platformData, setPlatformData] = useState<PlatformData[]>([]);
    const [seasonality, setSeasonality] = useState<SeasonalityData[]>([]);
    const [strategyRecommendation, setStrategyRecommendation] = useState<string>("");

    // NEW Option B state - consolidated filtered data
    interface FilteredData {
        kpis: FilteredKPI[];
        price_bands: { [key: string]: PriceBand[] };
        genre_lollipop: { [key: string]: GenreLollipop[] };
    }
    const [filteredData, setFilteredData] = useState<FilteredData | null>(null);
    const [heatmapData, setHeatmapData] = useState<HeatmapCell[]>([]);
    const [activeFilter, setActiveFilter] = useState<string>("All Games");

    // NEW: Zain Features state
    const [launchEcosystem, setLaunchEcosystem] = useState<LaunchEcosystem[]>([]);
    const [executionMultipliers, setExecutionMultipliers] = useState<ExecutionMultiplier[]>([]);
    const [proMatrix, setProMatrix] = useState<ProMatrix[]>([]);
    const [genreAnalysis, setGenreAnalysis] = useState<GenreAnalysis[]>([]);

    // Derived data based on active filter
    const filteredKpis = filteredData?.kpis || [];
    const priceBands = filteredData?.price_bands[activeFilter] || [];
    const genreLollipop = filteredData?.genre_lollipop[activeFilter] || [];

    useEffect(() => {
        // Core data
        fetch("/data/kpi_data.json").then(res => res.json()).then(setKpi).catch(e => console.error("KPI Error", e));
        fetch("/data/price_trends.json").then(res => res.json()).then(setPriceTrends).catch(e => console.error("Price Error", e));
        fetch("/data/ownership_tiers.json").then(res => res.json()).then(setOwnershipData).catch(e => console.error("Owner Error", e));
        fetch("/data/release_strategy.json").then(res => res.json()).then(setStrategyData).catch(e => console.error("Strategy Error", e));
        fetch("/data/strategy_scatter.json").then(res => res.json()).then(setStrategyScatter).catch(e => console.error("Scatter Error", e));

        // Genre analysis
        fetch("/data/genre_success.json").then(res => res.json()).then(setGenreSuccess).catch(e => console.error("GenreSuccess Error", e));
        fetch("/data/indie_matrix.json").then(res => res.json()).then(setIndieMatrix).catch(e => console.error("IndieMatrix Error", e));

        // Platform & Timing
        fetch("/data/platform_revenue.json").then(res => res.json()).then(setPlatformData).catch(e => console.error("Platform Error", e));
        fetch("/data/seasonality.json").then(res => res.json()).then(setSeasonality).catch(e => console.error("Seasonality Error", e));

        // NEW: Consolidated filtered data (All/Indie/Non-Indie)
        fetch("/data/filtered_data.json").then(res => res.json()).then(setFilteredData).catch(e => console.error("FilteredData Error", e));
        fetch("/data/heatmap.json").then(res => res.json()).then(setHeatmapData).catch(e => console.error("Heatmap Error", e));

        // NEW: Zain Features
        fetch("/data/launch_ecosystem.json").then(res => res.json()).then(setLaunchEcosystem).catch(e => console.error("LaunchEcosystem Error", e));
        fetch("/data/execution_multipliers.json").then(res => res.json()).then(setExecutionMultipliers).catch(e => console.error("Multipliers Error", e));
        fetch("/data/pro_matrix.json").then(res => res.json()).then(setProMatrix).catch(e => console.error("ProMatrix Error", e));
        fetch("/data/genre_analysis.json").then(res => res.json()).then(setGenreAnalysis).catch(e => console.error("GenreAnalysis Error", e));
    }, []);

    // Compute strategy aggregations - use useMemo to recalculate when data changes
    const { simMedian, gradMedian } = React.useMemo(() => {
        console.log("strategyData length:", strategyData.length);
        console.log("heatmapData loaded:", heatmapData.length, "cells"); // Future: Month×Price heatmap

        const simultaneousSales = strategyData.filter(d => d.Strategy === 'Simultaneous').map(d => d.Total_Sales);
        const gradualSales = strategyData.filter(d => d.Strategy === 'Gradual').map(d => d.Total_Sales);

        console.log("Simultaneous count:", simultaneousSales.length, "Gradual count:", gradualSales.length);

        const sortedSim = [...simultaneousSales].sort((a, b) => a - b);
        const sortedGrad = [...gradualSales].sort((a, b) => a - b);

        const simMed = sortedSim.length ? sortedSim[Math.floor(sortedSim.length / 2)] : 0;
        const gradMed = sortedGrad.length ? sortedGrad[Math.floor(sortedGrad.length / 2)] : 0;

        console.log("simMedian:", simMed, "gradMedian:", gradMed);

        return { simMedian: simMed, gradMedian: gradMed };
    }, [strategyData]);

    const handleStrategySelect = (value: string) => {
        if (value === "indie") {
            setStrategyRecommendation("Recommendation: Target a Gradual Release. Start on PC (Early Access) to build community. Avoid Q4 launch. Price at $19.99 to capture the 'Quality Indie' segment.");
        } else {
            setStrategyRecommendation("Recommendation: Leverage your budget for a simultaneous multi-platform launch IF you have existing IP. Otherwise, use the Gradual 'Scale-Up' strategy to minimize risk.");
        }
    };

    // Static Game Covers for Hero
    const products = [
        { title: "Hades", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7x.jpg" },
        { title: "Celeste", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1x7d.jpg" },
        { title: "Hollow Knight", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r77.jpg" },
        { title: "Stardew Valley", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/xpmm98drocehodoyynev.jpg" },
        { title: "Terraria", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1tnw.jpg" },
        { title: "Dead Cells", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co20r5.jpg" },
        { title: "Slay the Spire", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1imr.jpg" },
        { title: "Among Us", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co23x5.jpg" },
        { title: "Cuphead", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1q1f.jpg" },
        { title: "Undertale", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1tnq.jpg" },
        { title: "Limbo", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7i.jpg" },
        { title: "Inside", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7h.jpg" },
        { title: "Bastion", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7j.jpg" },
        { title: "Transistor", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7k.jpg" },
        { title: "Pyre", link: "#", thumbnail: "https://images.igdb.com/igdb/image/upload/t_cover_big/co1r7l.jpg" },
    ];

    if (!kpi) return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading Mission Data...</div>;

    return (
        <div className="bg-black min-h-screen text-white overflow-x-hidden">

            {/* 1. HERO SECTION */}
            <HeroParallax products={products} />

            <TracingBeam className="px-6">

                {/* 2. MARKET REALITY (KPIs) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-10 text-emerald-400">Mission Status: Market Reality</h2>
                    <BentoGrid>
                        <BentoGridItem
                            title="Total Games Analyzed"
                            description="Data collected from Steam & VGChartz."
                            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800 items-center justify-center text-4xl font-bold text-white">{kpi.total_games.toLocaleString()}</div>}
                            icon={<Gamepad2 className="h-4 w-4 text-neutral-500" />}
                            className="md:col-span-1"
                        />
                        <BentoGridItem
                            title="Indie Market Share"
                            description="The market is flooded. Standing out is harder than ever."
                            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800 items-center justify-center text-5xl font-extrabold text-red-500">{kpi.indie_share}%</div>}
                            icon={<Users className="h-4 w-4 text-neutral-500" />}
                            className="md:col-span-1"
                        />
                        <BentoGridItem
                            title="Median Indie Profit"
                            description="This is the reality for the 50th percentile."
                            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-emerald-900 to-black items-center justify-center text-4xl font-bold text-emerald-400">${kpi.median_indie_profit.toLocaleString()}</div>}
                            icon={<DollarSign className="h-4 w-4 text-neutral-500" />}
                            className="md:col-span-1"
                        />
                        <BentoGridItem
                            title="The Luxury Dominance"
                            description="Games priced over $60 hold significant share."
                            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-purple-900 to-black items-center justify-center text-4xl font-bold text-purple-400">{kpi.luxury_share}%</div>}
                            icon={<Trophy className="h-4 w-4 text-neutral-500" />}
                            className="md:col-span-3"
                        />
                    </BentoGrid>

                    {/* Filter Toggle (Option B) */}
                    <div className="mt-8 flex justify-center gap-4">
                        {filteredKpis.map((f) => (
                            <button
                                key={f.filter}
                                onClick={() => setActiveFilter(f.filter)}
                                className={`px-6 py-3 rounded-xl font-bold transition-all ${activeFilter === f.filter
                                    ? 'bg-emerald-600 text-white'
                                    : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                                    }`}
                            >
                                {f.filter} ({f.total.toLocaleString()})
                            </button>
                        ))}
                    </div>

                    {/* Active Filter KPIs */}
                    {filteredKpis.filter(f => f.filter === activeFilter).map((f) => (
                        <div key={f.filter} className="mt-6 grid grid-cols-3 gap-4 text-center">
                            <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                                <p className="text-2xl font-bold text-purple-400">${f.avg_price.toFixed(2)}</p>
                                <p className="text-sm text-neutral-500">Avg Price</p>
                            </div>
                            <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                                <p className="text-2xl font-bold text-emerald-400">${f.median_profit.toLocaleString()}</p>
                                <p className="text-sm text-neutral-500">Median Profit</p>
                            </div>
                            <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                                <p className="text-2xl font-bold text-yellow-400">{f.avg_score}%</p>
                                <p className="text-sm text-neutral-500">Avg Steam Score</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* 2b. PRICE BAND SUCCESS (Option B - Boxplot Style) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-cyan-400">Price Band Success</h2>
                    <p className="text-xl text-neutral-400 mb-10">Steam Score distribution by price tier - find the sweet spot.</p>

                    <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[500px]">
                        <h3 className="text-xl font-bold mb-4">Steam Score Distribution by Price Band</h3>
                        <ResponsiveContainer width="100%" height="85%">
                            <BarChart data={priceBands.map(b => ({
                                ...b,
                                boxBottom: b.q1,
                                boxHeight: b.q3 - b.q1,
                            }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="band" stroke="#888" />
                                <YAxis stroke="#888" domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                    labelStyle={{ color: "#fff" }}
                                    content={({ active, payload, label }) => {
                                        if (active && payload && payload.length) {
                                            const data = payload[0].payload;
                                            return (
                                                <div className="bg-neutral-900 p-3 rounded border border-neutral-700">
                                                    <p className="font-bold text-cyan-400">{label}</p>
                                                    <p className="text-sm">Min: {data.min?.toFixed(1)}%</p>
                                                    <p className="text-sm">Q1 (25th): {data.q1?.toFixed(1)}%</p>
                                                    <p className="text-sm font-bold text-white">Median: {data.median?.toFixed(1)}%</p>
                                                    <p className="text-sm">Q3 (75th): {data.q3?.toFixed(1)}%</p>
                                                    <p className="text-sm">Max: {data.max?.toFixed(1)}%</p>
                                                    <p className="text-xs text-neutral-500 mt-1">{data.count?.toLocaleString()} games</p>
                                                </div>
                                            );
                                        }
                                        return null;
                                    }}
                                />
                                {/* Invisible bar for Q1 offset */}
                                <Bar dataKey="boxBottom" stackId="box" fill="transparent" />
                                {/* Box showing IQR (Q1 to Q3) */}
                                <Bar dataKey="boxHeight" stackId="box" fill="#06b6d4" opacity={0.8} name="IQR" />
                            </BarChart>
                        </ResponsiveContainer>
                        <p className="text-sm text-neutral-500 text-center mt-2">Box shows Q1 (25th) to Q3 (75th) percentile. Hover for full stats.</p>
                    </div>

                    {/* Top 3 Price Bands Insight Cards */}
                    {priceBands.length > 0 && (
                        <div className="mt-6 grid grid-cols-3 gap-4">
                            {[...priceBands].sort((a, b) => b.median - a.median).slice(0, 3).map((band, idx) => (
                                <div key={band.band} className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800 text-center">
                                    <p className="text-2xl">{['🥇', '🥈', '🥉'][idx]}</p>
                                    <p className="text-xl font-bold text-cyan-400">{band.band}</p>
                                    <p className="text-sm text-neutral-400">Median: {band.median.toFixed(1)}%</p>
                                    <p className="text-xs text-neutral-500">{band.count.toLocaleString()} games</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* 3. THE ECONOMIC SQUEEZE (Charts) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-purple-400">The Economic Squeeze</h2>
                    <p className="text-xl text-neutral-400 mb-10">
                        Gamers are voting with their wallets for premium experiences.
                    </p>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[450px]">
                            <h3 className="text-xl font-bold mb-4">Average vs Median Price</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <LineChart data={priceTrends}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="year" stroke="#888" />
                                    <YAxis stroke="#888" />
                                    <Tooltip contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#fff" }} />
                                    <Legend wrapperStyle={{ paddingTop: "10px" }} />
                                    <Line type="monotone" dataKey="avg_price" stroke="#8b5cf6" strokeWidth={3} dot={false} name="Avg Price" />
                                    <Line type="monotone" dataKey="median_price" stroke="#10b981" strokeWidth={3} dot={false} name="Median Price" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[450px]">
                            <h3 className="text-xl font-bold mb-4">Market Share by Price Tier</h3>
                            <ResponsiveContainer width="100%" height="90%">
                                <AreaChart data={ownershipData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="year" stroke="#888" />
                                    <YAxis stroke="#888" tickFormatter={(val) => `${val}%`} />
                                    <Tooltip contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#fff" }} />
                                    <Area type="monotone" dataKey="Luxury" stackId="1" stroke="#a855f7" fill="#a855f7" name="Luxury (Top 10%)" />
                                    <Area type="monotone" dataKey="Premium" stackId="1" stroke="#6366f1" fill="#6366f1" name="Premium (Top 25%)" />
                                    <Area type="monotone" dataKey="Standard" stackId="1" stroke="#3b82f6" fill="#3b82f6" name="Standard (Top 50%)" />
                                    <Area type="monotone" dataKey="Budget" stackId="1" stroke="#ef4444" fill="#ef4444" name="Budget (Bottom 50%)" />
                                    <Legend wrapperStyle={{ paddingTop: "10px" }} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* 4. GENRE STRATEGY - NEW: Proper bar chart with genre names */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-yellow-400">Genre Strategy</h2>
                    <p className="text-xl text-neutral-400 mb-10">Which genres give indie developers the best chance of success?</p>

                    <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[500px]">
                        <h3 className="text-xl font-bold mb-4">Success Rate by Genre (% of games with 70%+ Steam Rating)</h3>
                        <ResponsiveContainer width="100%" height="85%">
                            <BarChart data={genreSuccess} layout="vertical" margin={{ left: 80 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis type="number" stroke="#888" tickFormatter={(val) => `${val}%`} domain={[0, 50]} />
                                <YAxis type="category" dataKey="genre" stroke="#888" width={75} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                    labelStyle={{ color: "#fff" }}
                                    formatter={(value: number, name: string) => {
                                        if (name === 'success_rate') return [`${value.toFixed(1)}%`, 'Success Rate'];
                                        if (name === 'mean_profit') return [`$${value.toLocaleString()}`, 'Avg Profit'];
                                        return [value, name];
                                    }}
                                />
                                <Bar dataKey="success_rate" fill="#fbbf24" name="success_rate" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Top 3 Genre Insight Cards */}
                    {genreLollipop.length > 0 && (
                        <div className="mt-6 grid grid-cols-3 gap-4">
                            {genreLollipop.slice(0, 3).map((g, idx) => (
                                <div key={g.genre} className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800 text-center">
                                    <p className="text-2xl">{['🥇', '🥈', '🥉'][idx]}</p>
                                    <p className="text-xl font-bold text-yellow-400">{g.genre}</p>
                                    <p className="text-sm text-neutral-400">Success: {g.success_rate.toFixed(1)}%</p>
                                    <p className="text-xs text-emerald-400">Median Profit: ${g.median_profit.toLocaleString()}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* 4b. INDIE MATRIX (Option B - Competition vs Success with Quadrants) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-pink-400">Indie Opportunity Matrix</h2>
                    <p className="text-xl text-neutral-400 mb-10">Competition vs Success Rate - find your safe haven.</p>

                    <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[500px]">
                        <ResponsiveContainer width="100%" height="90%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                                <CartesianGrid stroke="#333" />
                                <XAxis
                                    type="number"
                                    dataKey="competition"
                                    name="Competition"
                                    stroke="#888"
                                    domain={[0, 'auto']}
                                    label={{ value: 'Competition (Market Saturation %)', position: 'bottom', fill: '#888', offset: 0 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="success_rate"
                                    name="Success Rate"
                                    stroke="#888"
                                    domain={[0, 'auto']}
                                    tickFormatter={(val) => `${val}%`}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                    labelStyle={{ color: "#fff" }}
                                    formatter={(value: number, name: string) => {
                                        if (name === 'success_rate') return [`${value.toFixed(1)}%`, 'Success Rate'];
                                        if (name === 'competition') return [`${value.toFixed(1)}%`, 'Competition'];
                                        return [value, name];
                                    }}
                                />
                                <Scatter
                                    name="Genres"
                                    data={indieMatrix}
                                    fill="#ec4899"
                                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                    shape={(props: any) => {
                                        const { cx, cy, payload } = props;
                                        const color = payload.market_type === 'Safe Haven' ? '#10b981'
                                            : payload.market_type === 'Viable' ? '#8b5cf6'
                                                : payload.market_type === 'Crowded' ? '#f59e0b'
                                                    : '#ef4444';
                                        return (
                                            <g>
                                                <circle cx={cx} cy={cy} r={8} fill={color} fillOpacity={0.8} />
                                                <text x={cx} y={cy - 12} textAnchor="middle" fill="#fff" fontSize={10}>{payload.genre}</text>
                                            </g>
                                        );
                                    }}
                                />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Legend for Market Types */}
                    <div className="mt-4 flex justify-center gap-6 flex-wrap">
                        <span className="flex items-center gap-2"><span className="w-4 h-4 rounded-full bg-emerald-500"></span> Safe Haven</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-4 rounded-full bg-purple-500"></span> Viable</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-4 rounded-full bg-yellow-500"></span> Crowded</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-4 rounded-full bg-red-500"></span> Trap</span>
                    </div>
                </div>

                {/* 4c. PLATFORM & TIMING - NEW SECTION */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-blue-400">Platform & Launch Timing</h2>
                    <p className="text-xl text-neutral-400 mb-10">Where and when to launch for maximum impact.</p>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Platform Revenue */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[450px]">
                            <h3 className="text-xl font-bold mb-4">Platform Revenue Comparison</h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <BarChart data={platformData} margin={{ bottom: 50 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="platform" stroke="#888" angle={-25} textAnchor="end" height={60} tick={{ fontSize: 11 }} />
                                    <YAxis stroke="#888" tickFormatter={(val) => `$${val}M`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                        labelStyle={{ color: "#fff" }}
                                        formatter={(value: number) => [`$${value.toFixed(2)}M`, 'Median Revenue']}
                                    />
                                    <Bar dataKey="median_revenue" fill="#3b82f6" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Seasonality */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[450px]">
                            <h3 className="text-xl font-bold mb-4">Launch Window Analysis</h3>
                            <ResponsiveContainer width="100%" height="75%">
                                <LineChart data={seasonality}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="quarter" stroke="#888" />
                                    <YAxis stroke="#888" />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                        labelStyle={{ color: "#fff" }}
                                    />
                                    <Legend wrapperStyle={{ paddingTop: "10px" }} />
                                    <Line type="monotone" dataKey="traffic" stroke="#10b981" strokeWidth={3} name="Player Traffic" />
                                    <Line type="monotone" dataKey="competition" stroke="#ef4444" strokeWidth={3} strokeDasharray="5 5" name="Competition Risk" />
                                </LineChart>
                            </ResponsiveContainer>
                            <p className="text-center text-sm text-neutral-500 mt-2">⚠️ Q4 = High traffic but brutal competition. Launch in Q1 for best odds.</p>
                        </div>
                    </div>
                </div>

                {/* 4d. LAUNCH ECOSYSTEM (Zain's Research - For Gradual Releases) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-orange-400">Launch Ecosystem Strategy</h2>
                    <p className="text-xl text-neutral-400 mb-10">For gradual releases - which platform ecosystem to launch on FIRST (revenue distribution shown).</p>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Launch Ecosystem Chart */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[520px] flex flex-col">
                            <h3 className="text-xl font-bold mb-2">Median Revenue by Launch Platform</h3>
                            <p className="text-xs text-neutral-500 mb-4">Best first platform for gradual releases</p>
                            <div className="flex-1">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={[...launchEcosystem].sort((a, b) => b.median - a.median)}
                                        layout="vertical"
                                        margin={{ left: 0, right: 30, top: 0, bottom: 0 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={true} vertical={false} />
                                        <XAxis type="number" stroke="#888" domain={[0, 2]} tickFormatter={(v) => `$${v}M`} axisLine={false} />
                                        <YAxis type="category" dataKey="launch_type" stroke="#888" width={130} tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <Tooltip formatter={(v: number) => [`$${v.toFixed(2)}M`, 'Median']} contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444" }} />
                                        <Bar dataKey="median" fill="#f97316" radius={[0, 4, 4, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Execution Multipliers */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[450px]">
                            <h3 className="text-xl font-bold mb-4">Execution Multipliers (Traffic Lift)</h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <BarChart data={executionMultipliers}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="feature" stroke="#888" />
                                    <YAxis stroke="#888" domain={[0, 6]} tickFormatter={(val) => `${val}x`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }}
                                        formatter={(value: number) => [`${value}x baseline`, 'Traffic Lift']}
                                    />
                                    <Bar dataKey="traffic_lift">
                                        {executionMultipliers.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.category === 'Social' ? '#3b82f6' : '#10b981'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                            <p className="text-sm text-neutral-500 text-center">🔵 Social features | 🟢 Reach features</p>
                        </div>
                    </div>

                    {/* Insight Cards */}
                    <div className="mt-6 grid grid-cols-3 gap-4">
                        <div className="bg-gradient-to-br from-orange-900/50 to-neutral-900 p-4 rounded-xl border border-orange-800 text-center">
                            <p className="text-3xl font-bold text-orange-400">PC</p>
                            <p className="text-sm text-neutral-400">Best Launch Type</p>
                            <p className="text-xs text-orange-300">$1.68M median</p>
                        </div>
                        <div className="bg-gradient-to-br from-blue-900/50 to-neutral-900 p-4 rounded-xl border border-blue-800 text-center">
                            <p className="text-3xl font-bold text-blue-400">5.5x</p>
                            <p className="text-sm text-neutral-400">Multiplayer Boost</p>
                            <p className="text-xs text-blue-300">vs single-player</p>
                        </div>
                        <div className="bg-gradient-to-br from-emerald-900/50 to-neutral-900 p-4 rounded-xl border border-emerald-800 text-center">
                            <p className="text-3xl font-bold text-emerald-400">3.2x</p>
                            <p className="text-sm text-neutral-400">Global Languages</p>
                            <p className="text-xs text-emerald-300">10+ languages boost</p>
                        </div>
                    </div>
                </div>

                {/* 4e. PRO DOMINANCE MATRIX - Market Structure Matrix (Based on genre_analysis.json) */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-amber-400">Market Structure Matrix: Potential vs. Safety</h2>
                    <p className="text-xl text-neutral-400 mb-10">For high-budget studios: Safety (Median Traffic) vs Potential (Total Market Traffic). Color indicates Monopoly Risk (Inequality Ratio).</p>

                    <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[700px]">
                        {genreAnalysis.length > 0 && (() => {
                            // Prepare data exactly as in the notebook
                            const xData = genreAnalysis.map(d => d.Median_Traffic);
                            const yData = genreAnalysis.map(d => d.Total_Traffic);
                            // Scale sizes properly - Plotly expects sizes in pixels, scale Game_Count appropriately
                            const minSize = 10;
                            const maxSize = 50;
                            const minCount = Math.min(...genreAnalysis.map(d => d.Game_Count));
                            const maxCount = Math.max(...genreAnalysis.map(d => d.Game_Count));
                            const sizes = genreAnalysis.map(d => {
                                if (maxCount === minCount) return (minSize + maxSize) / 2;
                                return minSize + ((d.Game_Count - minCount) / (maxCount - minCount)) * (maxSize - minSize);
                            });
                            const colors = genreAnalysis.map(d => d.Inequality_Ratio);
                            const texts = genreAnalysis.map(d => d.Genre);

                            // Calculate TRUE medians for reference lines (to match notebook logic)
                            const sortedSafety = [...xData].sort((a, b) => a - b);
                            const sortedPotential = [...yData].sort((a, b) => a - b);
                            const medianSafety =
                                sortedSafety.length % 2 === 1
                                    ? sortedSafety[(sortedSafety.length - 1) / 2]
                                    : (sortedSafety[sortedSafety.length / 2 - 1] +
                                       sortedSafety[sortedSafety.length / 2]) / 2;
                            const medianPotential =
                                sortedPotential.length % 2 === 1
                                    ? sortedPotential[(sortedPotential.length - 1) / 2]
                                    : (sortedPotential[sortedPotential.length / 2 - 1] +
                                       sortedPotential[sortedPotential.length / 2]) / 2;

                            const plotData = [
                                {
                                    x: xData,
                                    y: yData,
                                    mode: 'markers+text',
                                    type: 'scatter',
                                    text: texts,
                                    textposition: 'top center',
                                    textfont: {
                                        size: 11,
                                        color: '#000'
                                    },
                                    marker: {
                                        size: sizes,
                                        color: colors,
                                        colorscale: 'RdYlGn_r', // Red-Yellow-Green reversed
                                        showscale: true,
                                        colorbar: {
                                            title: 'Monopoly Risk',
                                            titlefont: { size: 12 },
                                            tickfont: { size: 10 }
                                        },
                                        line: {
                                            color: 'rgba(0,0,0,0.3)',
                                            width: 1
                                        },
                                        opacity: 0.8,
                                        sizemode: 'diameter'
                                    },
                                    customdata: genreAnalysis.map(d => [
                                        d.Median_Traffic,
                                        d.Total_Traffic,
                                        d.Game_Count,
                                        d.Inequality_Ratio.toFixed(1)
                                    ]),
                                    hovertemplate: 
                                        '<b>%{text}</b><br>' +
                                        'Median Traffic: %{customdata[0]:,}<br>' +
                                        'Total Traffic: %{customdata[1]:,}<br>' +
                                        'Game Count: %{customdata[2]}<br>' +
                                        'Inequality Ratio: %{customdata[3]}<br>' +
                                        '<extra></extra>',
                                    name: 'Genres'
                                }
                            ];

                            // Calculate values for annotations and reference lines
                            const maxY = Math.max(...yData);
                            const minY = Math.min(...yData);

                            const layout = {
                                title: {
                                    text: 'RPG is Your Best Bet (Potential vs. Safety)<br><sup>Bubble size = Game Count, color = Monopoly Risk</sup>',
                                    font: { size: 20 },
                                    x: 0.5,
                                    xanchor: 'center'
                                },
                                xaxis: {
                                    title: {
                                        text: 'SAFETY: Median Player Traffic (Log Scale)'
                                    },
                                    type: 'log'
                                    // Let Plotly auto-scale (like px.scatter does) - no manual range
                                },
                                yaxis: {
                                    title: {
                                        text: 'POTENTIAL: Total Market Traffic (Log Scale)'
                                    },
                                    type: 'log'
                                    // Let Plotly auto-scale (like px.scatter does) - no manual range
                                },
                                template: 'plotly_white',
                                height: 700,
                                width: 1000,
                                title_font_size: 20,
                                shapes: [
                                    // Vertical Line (Median Safety) - using paper coordinates for y
                                    {
                                        type: 'line',
                                        x0: medianSafety,
                                        x1: medianSafety,
                                        y0: 0,
                                        y1: 1,
                                        yref: 'paper',
                                        line: { color: 'gray', dash: 'dash', width: 1 }
                                    },
                                    // Horizontal Line (Median Potential) - using paper coordinates for x
                                    {
                                        type: 'line',
                                        x0: 0,
                                        x1: 1,
                                        xref: 'paper',
                                        y0: medianPotential,
                                        y1: medianPotential,
                                        line: { color: 'gray', dash: 'dash', width: 1 }
                                    }
                                ],
                                annotations: [
                                    // GOLD MINE annotation (paper coordinates, like in notebook)
                                    {
                                        x: 0.95,
                                        y: 0.95,
                                        xref: 'paper',
                                        yref: 'paper',
                                        xanchor: 'right',
                                        yanchor: 'top',
                                        text: '<b>GOLD MINE</b><br>(High Safety, High Potential)',
                                        showarrow: false,
                                        font: { size: 12, color: 'green' }
                                    },
                                    // GRAVEYARD annotation (paper coordinates, like in notebook)
                                    {
                                        x: 0.08,
                                        y: 0.08,
                                        xref: 'paper',
                                        yref: 'paper',
                                        xanchor: 'left',
                                        yanchor: 'bottom',
                                        text: '<b>GRAVEYARD</b><br>(Low Safety, Low Potential)',
                                        showarrow: false,
                                        font: { size: 12, color: 'red' }
                                    }
                                ]
                            };

                            const config = {
                                displayModeBar: true,
                                displaylogo: false,
                                responsive: true,
                                toImageButtonOptions: {
                                    format: 'png',
                                    filename: 'market_structure_matrix',
                                    height: 700,
                                    width: 1000,
                                    scale: 2
                                }
                            };

                            return (
                                <Plot
                                    data={plotData}
                                    layout={layout}
                                    config={config}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            );
                        })()}
                        {genreAnalysis.length === 0 && (
                            <div className="flex items-center justify-center h-full text-neutral-400">
                                Loading chart data...
                            </div>
                        )}
                    </div>

                    {/* Legend and Annotations */}
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                            <h4 className="font-bold text-white mb-3">Monopoly Risk (Inequality Ratio)</h4>
                            <div className="space-y-2 text-sm">
                                <div className="flex items-center gap-2">
                                    <span className="w-4 h-4 rounded-full bg-emerald-500"></span>
                                    <span>Low Risk (&lt;50) - Fair Economy</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="w-4 h-4 rounded-full bg-lime-500"></span>
                                    <span>Medium Risk (50-100) - Moderate Inequality</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="w-4 h-4 rounded-full bg-yellow-500"></span>
                                    <span>High Risk (100-150) - Winner-Take-All</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="w-4 h-4 rounded-full bg-orange-500"></span>
                                    <span>Very High Risk (150-200) - Casino Market</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="w-4 h-4 rounded-full bg-red-500"></span>
                                    <span>Extreme Risk (&gt;200) - Monopoly Market</span>
                                </div>
                            </div>
                        </div>
                        <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                            <h4 className="font-bold text-white mb-3">Key Insights</h4>
                            <div className="space-y-2 text-sm text-neutral-300">
                                <p>• <span className="text-emerald-400 font-bold">RPG/Massively Multiplayer</span>: High Safety with substantial market potential</p>
                                <p>• <span className="text-red-400 font-bold">Action/Adventure</span>: High Total Traffic but very high Inequality (Winner-Take-All)</p>
                                <p>• <span className="text-yellow-400 font-bold">Racing/Education</span>: Lower competition but limited market size</p>
                                <p className="mt-3 text-xs text-neutral-500">Target genres with balanced Safety & Potential and low Inequality Ratio</p>
                            </div>
                        </div>
                    </div>
                </div>


                {/* 5. THE GRADUAL RELEASE SECRET */}
                <div className="max-w-[95%] mx-auto py-20">
                    <h2 className="text-4xl md:text-5xl font-bold mb-6 text-emerald-400">The "Gradual Release" Secret</h2>

                    {/* Key Insight Card */}
                    <div className="w-full rounded-2xl p-10 text-xl md:text-4xl font-bold text-white bg-gradient-to-br from-emerald-900 to-black border border-emerald-500/30 mb-10">
                        <p>Gradual Releases show <span className="text-emerald-400">{(gradMedian / (simMedian || 1)).toFixed(1)}x</span> higher median revenue.</p>
                        <p className="text-lg mt-4 text-neutral-400">Simultaneous: {simMedian.toFixed(2)}M | Gradual: {gradMedian.toFixed(2)}M</p>
                        <div className="mt-8 text-base font-normal text-neutral-300">
                            Strategy: Start on PC, then port to consoles later.
                        </div>
                    </div>

                    {/* Charts Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Bar Chart */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[400px]">
                            <h3 className="text-xl font-bold mb-4">Sales Comparison</h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <BarChart data={[
                                    { name: 'Simultaneous', sales: simMedian },
                                    { name: 'Gradual', sales: gradMedian },
                                ]}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="name" stroke="#888" />
                                    <YAxis stroke="#888" tickFormatter={(val) => `${val.toFixed(1)}M`} />
                                    <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#fff" }} />
                                    <Bar dataKey="sales" fill="#10b981" barSize={80} name="Median Sales (Millions)" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Scatter Chart */}
                        <div className="bg-neutral-900/50 p-6 rounded-3xl border border-neutral-800 h-[400px]">
                            <h3 className="text-xl font-bold mb-4">Strategy Matrix</h3>
                            <ResponsiveContainer width="100%" height="85%">
                                <ScatterChart>
                                    <CartesianGrid stroke="#333" />
                                    <XAxis type="number" dataKey="Release_Span" name="Release Span (Years)" stroke="#888" />
                                    <YAxis type="number" dataKey="Total_Sales" name="Total Sales (M)" stroke="#888" tickFormatter={(val) => `${val.toFixed(1)}M`} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #444", color: "#fff" }} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#fff" }} />
                                    <Scatter name="Games" data={strategyScatter} fill="#8b5cf6" fillOpacity={0.7} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* 6. STRATEGY ENGINE */}
                <div className="max-w-[95%] mx-auto py-20 relative min-h-[800px] mt-20">
                    <BackgroundBeams className="opacity-20" />
                    <div className="relative z-10">
                        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-center text-white">Strategy Engine</h2>
                        <p className="text-xl text-neutral-400 text-center mb-10">Select your profile to get personalized recommendations</p>

                        <HoverEffect items={[
                            { title: "Bootstrapped Indie", description: "Low budget. High passion.", value: "indie" },
                            { title: "Venture Backed / Pro", description: "Significant budget. Need scale.", value: "pro" },
                        ]} onSelect={handleStrategySelect} />

                        {strategyRecommendation && (
                            <div className="mt-10 space-y-6">
                                {/* Main Recommendation Card */}
                                <div className="p-8 rounded-3xl bg-gradient-to-br from-emerald-900/30 to-black border border-emerald-500/50">
                                    <h3 className="text-2xl font-bold text-emerald-400 mb-4">📋 Your Strategy</h3>
                                    <TextGenerateEffect words={strategyRecommendation} />
                                </div>

                                {/* Key Action Cards */}
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="p-6 rounded-2xl bg-neutral-900/50 border border-neutral-800">
                                        <p className="text-3xl mb-2">🎯</p>
                                        <h4 className="font-bold text-white mb-2">Release Strategy</h4>
                                        <p className="text-sm text-neutral-400">
                                            {strategyRecommendation.includes("Gradual")
                                                ? "Gradual Release - Start PC, port to consoles later"
                                                : "Multi-Platform - Launch everywhere simultaneously"}
                                        </p>
                                    </div>
                                    <div className="p-6 rounded-2xl bg-neutral-900/50 border border-neutral-800">
                                        <p className="text-3xl mb-2">💰</p>
                                        <h4 className="font-bold text-white mb-2">Price Point</h4>
                                        <p className="text-sm text-neutral-400">
                                            {strategyRecommendation.includes("$19.99")
                                                ? "$19.99 - 'Quality Indie' sweet spot"
                                                : "$39.99-$59.99 - Full price justified by scope"}
                                        </p>
                                    </div>
                                    <div className="p-6 rounded-2xl bg-neutral-900/50 border border-neutral-800">
                                        <p className="text-3xl mb-2">📅</p>
                                        <h4 className="font-bold text-white mb-2">Launch Timing</h4>
                                        <p className="text-sm text-neutral-400">
                                            {strategyRecommendation.includes("Q4")
                                                ? "Avoid Q4 - Too much AAA competition"
                                                : "Q4 viable with marketing budget"}
                                        </p>
                                    </div>
                                </div>

                                {/* Genre Recommendations based on selection */}
                                <div className="p-6 rounded-2xl bg-neutral-900/50 border border-neutral-800">
                                    <h4 className="font-bold text-white mb-4">🎮 Recommended Genres</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {strategyRecommendation.includes("Gradual") ? (
                                            <>
                                                <span className="px-3 py-1 rounded-full bg-emerald-900/50 text-emerald-400 text-sm">Roguelike</span>
                                                <span className="px-3 py-1 rounded-full bg-emerald-900/50 text-emerald-400 text-sm">Simulation</span>
                                                <span className="px-3 py-1 rounded-full bg-emerald-900/50 text-emerald-400 text-sm">Strategy</span>
                                                <span className="px-3 py-1 rounded-full bg-emerald-900/50 text-emerald-400 text-sm">Puzzle</span>
                                            </>
                                        ) : (
                                            <>
                                                <span className="px-3 py-1 rounded-full bg-purple-900/50 text-purple-400 text-sm">Action RPG</span>
                                                <span className="px-3 py-1 rounded-full bg-purple-900/50 text-purple-400 text-sm">Open World</span>
                                                <span className="px-3 py-1 rounded-full bg-purple-900/50 text-purple-400 text-sm">Multiplayer</span>
                                                <span className="px-3 py-1 rounded-full bg-purple-900/50 text-purple-400 text-sm">Sports</span>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

            </TracingBeam>

        </div>
    );
}
