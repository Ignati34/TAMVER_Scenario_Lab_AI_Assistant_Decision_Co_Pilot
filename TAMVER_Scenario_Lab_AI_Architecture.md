# TAMVER Scenario Lab AI — архитектура

## 1. Назначение

AI-ассистент не принимает решение вместо пользователя. Он:

1. интерпретирует цель;
2. генерирует допустимые действия;
3. прогоняет их через AIR;
4. проверяет устойчивость через Monte Carlo;
5. ранжирует варианты;
6. показывает причинные цепочки;
7. передает результат человеку на утверждение.

## 2. Контур

```text
User command
   ↓
Intent and objective profile
   ↓
Candidate decision generator
   ↓
AIR nonlinear simulation
   ↓
Monte Carlo robustness check
   ↓
Risk / stability / cost score
   ↓
Ranking + causal explanation
   ↓
Human approval
```

## 3. Функциональные блоки

- **Scenario Lab:** ручной what-if анализ.
- **Decision Co-Pilot:** автоматический поиск решений.
- **Stress Test:** распределение рисков и P95.
- **Centrality:** критические узлы системы.
- **Causal Paths:** наиболее сильные цепочки распространения.
- **Graph View:** визуальная карта узлов и решений.

## 4. Контроль качества

- автоматическая проверка спектральной устойчивости;
- нелинейное ограничение через `tanh`;
- confidence для каждой связи;
- uncertainty для каждого узла;
- бюджетное ограничение решений;
- отдельная оценка deterministic risk и stress risk P95;
- сохранение роли человека как decision owner.

## 5. Следующее расширение

- подключение `air_matrix_calculated_v0_3`;
- calibration registry;
- versioning коэффициентов;
- historical backtesting;
- Approval workflow;
- Decision Audit Report;
- SaaS multi-tenant layer.
