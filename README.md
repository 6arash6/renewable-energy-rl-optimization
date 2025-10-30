# Project Proposal: Reinforcement Learning for Optimizing Renewable Energy Systems

## 1. Project Title
**Dynamic Harmony: Reinforcement Learning for Adaptive Optimization in Hybrid Renewable Energy Ecosystems**

## 2. Project Goal
The goal is to develop and evaluate a reinforcement learning (RL) model that optimizes energy storage and dispatch in a simulated hybrid renewable energy system (combining solar panels, wind turbines, and battery storage) to reduce overall energy costs by at least 15% compared to baseline rule-based strategies, while maintaining grid stability and minimizing renewable energy curtailment.

**Success Metrics:**
- Primary: ≥15% cost reduction vs. baseline
- Secondary: 
  - Renewable energy utilization rate ≥85%
  - Grid stability variance <10%
  - Energy curtailment <5%
  - Battery cycle efficiency ≥90%

**Validation Period:** 3-month simulated period using historical data (Jan-Mar 2020)

## 3. Group Members
- **Arash**

## 4. Data Sources

### Selected Datasets

#### 4.1 Renewables.ninja Dataset
- **Description**: Simulated hourly renewable energy production data (wind and solar output) based on global weather models
- **Source**: [Renewables Ninja](https://www.renewables.ninja/)
- **Temporal Coverage**: 2015-2020
- **Spatial Resolution**: Customizable (European region focus)
- **Variables**: Solar irradiance (W/m²), wind speed (m/s), temperature (°C), power output (kW)

#### 4.2 ENTSO-E Transparency Platform Dataset
- **Description**: Historical electricity demand, pricing, and grid data across Europe
- **Source**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- **Temporal Coverage**: 2015-2020
- **Variables**: Electricity demand (kWh), day-ahead prices (€/kWh), actual generation, cross-border flows
- **API Access**: Available via REST API

#### 4.3 Kaggle Solar Power Generation Data
- **Description**: Solar plant generation and weather sensor data
- **Source**: [Kaggle Solar Power Generation](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Variables**: DC/AC power (kW), module temperature (°C), irradiation sensors
- **Use Case**: Validation and supplementary weather correlation

### Critical Variables & Data Types

| Variable | Type | Unit | Resolution | Role in RL |
|----------|------|------|------------|------------|
| Wind Speed | Float | m/s | Hourly | State input |
| Solar Irradiance | Float | W/m² | Hourly | State input |
| Power Output (Renewables) | Float | kW | Hourly | State input |
| Electricity Demand | Float | kWh | Hourly | State input |
| Electricity Price | Float | €/kWh | Hourly | Reward signal |
| Battery State of Charge | Float | % (0-100) | Continuous | State/Action |
| Temperature | Float | °C | Hourly | State input |
| Grid Import/Export | Float | kW | Hourly | Action output |

**Data Quality Considerations:**
- Expected missing data: <2% (handled via interpolation)
- Outlier detection: Statistical methods (IQR, Z-score)
- Validation: Cross-reference multiple sources

## 5. Analytics and AI Methods

### 5.1 Methodology Overview
The project models the hybrid energy system as a **Markov Decision Process (MDP)** where:
- **States (S)**: Battery SoC, renewable output, demand, prices, time of day
- **Actions (A)**: Battery charge/discharge rate, grid buy/sell quantity
- **Rewards (R)**: Negative cost (minimization objective) + stability penalties
- **Transition**: Deterministic for battery dynamics, stochastic for renewables/demand

### 5.2 Implementation Pipeline

#### Phase 1: Data Preprocessing (Weeks 1-3)
1. **Data Acquisition**: Download and merge datasets
2. **Cleaning**: Handle missing values (forward-fill for short gaps, interpolation for longer)
3. **Normalization**: Min-Max scaling for neural network inputs
4. **Feature Engineering**:
   - Lagged features (t-1, t-24 hours)
   - Rolling statistics (24h mean/std)
   - Cyclical encoding (hour, day, season)
   - Weather forecast errors (actual vs. predicted)
5. **Train/Test Split**: 80/20 temporal split (preserve time order)

#### Phase 2: Environment Development (Weeks 4-5)
- **Framework**: Custom OpenAI Gym environment
- **System Model**:
  ```
  Battery dynamics: SoC(t+1) = SoC(t) + η_charge * P_charge - P_discharge / η_discharge
  Energy balance: Demand = Renewables + Grid_import + Battery_discharge
  Constraints: SoC ∈ [20%, 95%], Grid_import ≤ Max_capacity
  ```
- **Reward Function**:
  ```
  R(t) = -[Cost_grid(t) + λ₁ * Penalty_curtailment + λ₂ * Penalty_stability]
  where Cost_grid = Price(t) * Grid_import(t)
  ```

#### Phase 3: RL Model Training (Weeks 6-10)

**Algorithm Selection:**
1. **Baseline**: Deep Q-Network (DQN)
   - Discrete action space (5 levels: heavy discharge → heavy charge)
   - Experience replay buffer (100k transitions)
   - Target network updates every 1000 steps
   
2. **Advanced**: Proximal Policy Optimization (PPO)
   - Continuous action space
   - Actor-Critic architecture (shared layers)
   - Clipped surrogate objective for stability

**Training Configuration:**
- Episodes: 1000 (each = 720 hours / 1 month)
- Learning rate: 1e-4 (Adam optimizer)
- Discount factor (γ): 0.99
- Exploration: ε-greedy (ε: 1.0 → 0.1 over 500 episodes)
- Batch size: 64

**Hyperparameter Tuning:**
- Grid search over: learning rate, network architecture, reward weights
- Validation: Rolling 3-month window

#### Phase 4: Baseline Comparisons (Weeks 11-12)

**Comparison Strategies:**
1. **Rule-Based Controller**: 
   - Charge when price < threshold
   - Discharge when price > threshold
   
2. **Time-of-Use (ToU) Optimizer**:
   - Predefined charge/discharge schedule based on typical patterns
   
3. **Model Predictive Control (MPC)**:
   - 24-hour forecast horizon
   - Linear optimization

**Evaluation Metrics:**
- **Economic**: Total cost, cost per kWh, ROI
- **Technical**: Energy efficiency, curtailment %, battery cycles
- **Reliability**: Demand satisfaction rate, grid stability variance
- **Environmental**: Renewable penetration rate, CO₂ avoided

#### Phase 5: Validation & Reporting (Weeks 13-14)
- Test on unseen 2020 Q2 data
- Sensitivity analysis (weather variability, price volatility)
- Statistical significance testing (paired t-test)
- Documentation and final presentation

### 5.3 Technical Stack

**Core Libraries:**
- **RL**: Stable-Baselines3, TensorFlow/PyTorch
- **Data**: Pandas, NumPy, Dask (for large datasets)
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Environment**: OpenAI Gym, Custom wrappers
- **Optimization**: SciPy, CVXPY (for MPC baseline)

**Computational Requirements:**
- GPU: NVIDIA GTX 1080 or equivalent (8GB VRAM)
- Training time estimate: 10-15 hours for DQN, 20-30 hours for PPO
- Storage: ~5GB for datasets, ~2GB for models

### 5.4 Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data quality issues | Medium | High | Multi-source validation, robust preprocessing |
| Model convergence failure | Medium | High | Simplified reward, curriculum learning |
| Computational constraints | Low | Medium | Cloud resources (Google Colab, AWS free tier) |
| Overfitting to historical data | High | Medium | Cross-validation, diverse test scenarios |
| Unrealistic simulation dynamics | Medium | High | Literature-based parameter validation |

## 6. Expected Outcomes & Deliverables

### Technical Deliverables:
1. **Trained RL Models**: DQN and PPO agents with saved weights
2. **Simulation Environment**: Reusable Gym environment code
3. **Evaluation Framework**: Scripts for metrics computation and visualization
4. **Comparative Analysis**: Performance report vs. baselines

### Documentation:
1. **Code Repository**: Well-documented GitHub repo with README
2. **Technical Report**: 15-20 pages covering methodology, results, analysis
3. **Presentation**: 15-minute final presentation with demo

### Knowledge Contributions:
- Insights on RL hyperparameter sensitivity in energy systems
- Recommendations for real-world deployment considerations
- Open-source toolkit for renewable energy optimization research

## 7. Timeline & Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-3 | Data acquisition & preprocessing | Clean datasets, EDA report |
| 4-5 | Environment development | Functional Gym environment |
| 6-8 | DQN implementation & training | Trained DQN model |
| 9-10 | PPO implementation & training | Trained PPO model |
| 11-12 | Baseline comparisons & evaluation | Comparative results |
| 13 | Validation & sensitivity analysis | Final metrics report |
| 14 | Documentation & presentation | Final report & presentation |

## 8. References

1. **Reinforcement Learning for Sustainable Energy: A Survey** - [arXiv:2407.18597](https://arxiv.org/html/2407.18597v1)
2. **Applications of reinforcement learning in energy systems** - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364032120309023)
3. **RL for Optimizing Renewable Energy Utilization in Buildings** - [MDPI Energies](https://www.mdpi.com/1996-1073/18/7/1724)
4. **Deep RL for Mobile Energy Storage Systems** - [MDPI Batteries](https://www.mdpi.com/2313-0105/9/4/219)
5. **Optimization of O&M of renewable systems by Deep RL** - [Renewable Energy](https://ideas.repec.org/a/eee/renene/v183y2022icp752-763.html)

---

**Project Alignment with Course Objectives:**
- ✅ Applies optimization/RL to renewable energy challenges
- ✅ Integrates multiple data sources for real-world scenarios
- ✅ Demonstrates measurable impact on sustainability metrics
- ✅ Balances technical rigor with practical feasibility
- ✅ Promotes reproducible research through open methodology
