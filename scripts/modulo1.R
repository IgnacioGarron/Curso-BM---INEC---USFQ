paquetes <- c("tidymodels", "tidyverse", "broom", "glmnet", "readxl")

suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(broom)
  library(readxl)
})

# tidymodels_prefer() resuelve conflictos de nombres a favor de tidymodels.
# Esto evita ambigüedades cuando varios paquetes tienen funciones parecidas.
tidymodels_prefer()
theme_set(theme_minimal(base_size = 12))

# Reproducibilidad
semilla_global <- 123
set.seed(semilla_global)

# Carpeta única para guardar archivos generados por el módulo.
output_dir <- file.path("output", "modulo1")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# sessionInfo()

set.seed(semilla_global)
n <- 1000

hogares <- tibble(
  id_hogar = 1:n,
  region = sample(c("Costa", "Sierra", "Amazonia", "Insular"), n, replace = TRUE, prob = c(.46, .42, .10, .02)),
  area = sample(c("Urbana", "Rural"), n, replace = TRUE, prob = c(.68, .32)),
  miembros = pmax(1, rpois(n, lambda = 3) + 1),
  edad_jefe = pmin(85, pmax(18, round(rnorm(n, mean = 46, sd = 14)))),
  educ_jefe = sample(c("Primaria", "Secundaria", "Superior"), n, replace = TRUE, prob = c(.38, .43, .19)),
  empleo_jefe = sample(c("Formal", "Informal", "Desempleado", "Inactivo"), n, replace = TRUE, prob = c(.36, .39, .08, .17)),
  vivienda_propia = sample(c("Sí", "No"), n, replace = TRUE, prob = c(.62, .38)),
  internet = sample(c("Sí", "No"), n, replace = TRUE, prob = c(.57, .43)),
  transferencias = rgamma(n, shape = 1.5, rate = 1 / 55)
) %>%
  mutate(
    rural = if_else(area == "Rural", 1, 0),
    educ_num = case_when(
      educ_jefe == "Primaria" ~ 6,
      educ_jefe == "Secundaria" ~ 12,
      educ_jefe == "Superior" ~ 17
    ),
    empleo_efecto = case_when(
      empleo_jefe == "Formal" ~ 280,
      empleo_jefe == "Informal" ~ 90,
      empleo_jefe == "Desempleado" ~ -160,
      empleo_jefe == "Inactivo" ~ -70
    ),
    region_efecto = case_when(
      region == "Costa" ~ 35,
      region == "Sierra" ~ 10,
      region == "Amazonia" ~ -55,
      region == "Insular" ~ 170
    ),
    ingreso_pc = 50 + 42 * educ_num + empleo_efecto + region_efecto -
      38 * miembros - 95 * rural + 1.8 * edad_jefe + .55 * transferencias +
      rnorm(n, 0, 120),
    ingreso_pc = pmax(30, round(ingreso_pc, 2)),
    pobre = factor(if_else(ingreso_pc < 10 * 31, "Pobre", "No pobre"), levels = c("No pobre", "Pobre"))
  ) %>%
  select(-rural, -educ_num, -empleo_efecto, -region_efecto) %>%
  mutate(
    # Introducimos faltantes controlados en predictores para practicar imputación.
    internet = if_else(runif(n()) < .04, NA_character_, internet),
    empleo_jefe = if_else(runif(n()) < .03, NA_character_, empleo_jefe),
    transferencias = if_else(runif(n()) < .03, NA_real_, transferencias),
    edad_jefe = if_else(runif(n()) < .02, NA_real_, edad_jefe)
  )

dir.create("data", showWarnings = FALSE)
write_csv(hogares, "data/hogares_modulo1.csv")

hogares %>% glimpse()

resumen_hogares <- hogares %>%
  summarise(
    n = n(),
    ingreso_promedio = mean(ingreso_pc),
    ingreso_mediano = median(ingreso_pc),
    tasa_pobreza = mean(pobre == "Pobre")
  )
# Cerca de salario medio de Ecuador 528$ y tasa ~20%

missing_hogares <- hogares %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  filter(n_missing > 0)

write_csv(resumen_hogares, file.path(output_dir, "resumen_hogares.csv"))
write_csv(missing_hogares, file.path(output_dir, "missing_hogares.csv"))

resumen_hogares
missing_hogares

# Función auxiliar para imputar categorias con la moda.
moda <- function(x) {
  valores <- x[!is.na(x)]
  if (length(valores) == 0) return(NA)
  valores[which.max(tabulate(match(valores, unique(valores))))]
}

hogares_imputados <- hogares %>%
  mutate(
    across(where(is.numeric), ~ replace_na(.x, mean(.x, na.rm = TRUE))),
    across(where(~ is.character(.x) || is.factor(.x)), ~ replace_na(.x, moda(.x)))
  )

write_csv(hogares_imputados, file.path(output_dir, "hogares_imputados_media_moda.csv"))

hogares_imputados %>% summarise(across(everything(), ~ sum(is.na(.x))))

variables_numericas_modelo <- c("miembros", "edad_jefe", "transferencias")

parametros_normalización <- hogares_imputados %>%
  summarise(across(
    all_of(variables_numericas_modelo),
    list(media = mean, desviacion = sd),
    .names = "{.col}_{.fn}"
  )) %>%
  pivot_longer(
    everything(),
    names_to = c("variable", "estadistico"),
    names_pattern = "(.+)_(media|desviacion)",
    values_to = "valor"
  ) %>%
  pivot_wider(names_from = estadistico, values_from = valor)

hogares_normalizados <- hogares_imputados %>%
  mutate(across(
    all_of(variables_numericas_modelo),
    ~ (.x - mean(.x)) / sd(.x),
    .names = "{.col}_z"
  ))

write_csv(parametros_normalización, file.path(output_dir, "parametros_normalización.csv"))
write_csv(hogares_normalizados, file.path(output_dir, "hogares_normalizados.csv"))

parametros_normalización
hogares_normalizados %>%
  select(id_hogar, ingreso_pc, pobre, ends_with("_z")) %>%
  glimpse()

hogares %>%
  ggplot(aes(x = ingreso_pc, fill = pobre)) +
  geom_histogram(bins = round(nrow(hogares)^(1 / 2), 0), alpha = .75, color = "white") +
  facet_wrap(~ area) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "Ingreso per cápita", y = "Hogares", fill = "Condición", title = "Distribución de ingreso por zona")

# Proporción real de pobreza en todos los datos.
prop_pobre_real <- mean(hogares$pobre == "Pobre")

# 1) Partición simple.
set.seed(semilla_global)
split_simple <- initial_split(hogares, prop = 0.80)
train_simple <- training(split_simple)
test_simple <- testing(split_simple)

# 2) Partición estratificada.
set.seed(semilla_global)
split_hogares <- initial_split(hogares, prop = 0.80, strata = pobre)

train <- training(split_hogares)
test <- testing(split_hogares)

# 3) K-fold sin estratos y 4) K-fold con estratos.
set.seed(semilla_global)
folds_simple <- vfold_cv(hogares, v = 5)

set.seed(semilla_global)
folds_strata <- vfold_cv(hogares, v = 5, strata = pobre)

# 5) Bootstrap sin estratos y 6) Bootstrap con estratos.
set.seed(semilla_global)
boots_simple <- bootstraps(hogares, times = 5)

set.seed(semilla_global)
boots_strata <- bootstraps(hogares, times = 5, strata = pobre)

# Resumen básico: proporción de pobreza en cada muestra.
resumen_partición <- tibble(
  particion = c(
    "Datos completos",
    "Initial split simple - Train",
    "Initial split simple - Test",
    "Initial split con strata - Train",
    "Initial split con strata - Test",
    "K-fold simple - Fold 1",
    "K-fold con strata - Fold 1",
    "Bootstrap simple - Muestra 1",
    "Bootstrap con strata - Muestra 1"
  ),
  prop_pobre = c(
    prop_pobre_real,
    mean(train_simple$pobre == "Pobre"),
    mean(test_simple$pobre == "Pobre"),
    mean(train$pobre == "Pobre"),
    mean(test$pobre == "Pobre"),
    mean(assessment(folds_simple$splits[[1]])$pobre == "Pobre"),
    mean(assessment(folds_strata$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_simple$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_strata$splits[[1]])$pobre == "Pobre")
  )
) %>%
  mutate(
    prop_real = prop_pobre_real,
    diferencia_vs_real = prop_pobre - prop_real
  )

write_csv(resumen_partición, file.path(output_dir, "resumen_partición.csv"))

resumen_partición

# La receta aprende imputación, dummies y normalización solo con train.

# Partición estratificada.
set.seed(semilla_global)
split_hogares <- initial_split(hogares, prop = 0.80, strata = pobre)

train <- training(split_hogares)
test <- testing(split_hogares)

# receta

receta_ols <- recipe(ingreso_pc ~ region + area + miembros + edad_jefe + educ_jefe + empleo_jefe +
  vivienda_propia + internet + transferencias,
data = train
) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# parsnip declara OLS y workflow une preparación + modelo.
modelo_ols <- linear_reg() %>%
  set_engine("lm")

# workflow
wf_ols <- workflow() %>%
  add_recipe(receta_ols) %>%
  add_model(modelo_ols)

wf_ols

# fit() entrena el workflow completo; predict() evalúa en test.
ajuste_ols <- fit(wf_ols, data = train)

pred_ols <- predict(ajuste_ols, new_data = test) %>%
  bind_cols(test %>% select(ingreso_pc))

head(pred_ols)

# yardstick
metr <- metric_set(rmse, rsq, mae)

metricas_ols <- metr(
  pred_ols,
  truth = ingreso_pc,
  estimate = .pred
)

# Guardar output
write_csv(metricas_ols, file.path(output_dir, "metricas_ols.csv"))

# Partición estratificada.
set.seed(semilla_global)
split_hogares <- initial_split(hogares, prop = 0.80, strata = pobre)

train <- training(split_hogares)
test <- testing(split_hogares)

# Receta de clasificación: misma preparación, distinto objetivo.
receta_logit <- recipe(pobre ~ region + area + miembros + edad_jefe + educ_jefe + empleo_jefe +
  vivienda_propia + internet + transferencias,
data = train
) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

modelo_logit <- logistic_reg() %>%
  set_engine("glm")

wf_logit <- workflow() %>%
  add_recipe(receta_logit) %>%
  add_model(modelo_logit)

wf_logit

ajuste_logit <- fit(wf_logit, data = train)

pred_logit <- predict(ajuste_logit, new_data = test, type = "prob") %>%
  bind_cols(predict(ajuste_logit, new_data = test, type = "class")) %>%
  bind_cols(test %>% select(pobre))

head(pred_logit)

# metric_set
metr <- metric_set(
  accuracy,
  precision,
  recall,
  f_meas,
  roc_auc
)

metricas_logit <- metr(
  pred_logit,
  truth = pobre,
  estimate = .pred_class,
  .pred_Pobre,
  event_level = "second"
)

matriz_confusion <- conf_mat(
  pred_logit,
  truth = pobre,
  estimate = .pred_class
)

write_csv(metricas_logit, file.path(output_dir, "metricas_logit.csv"))

metricas_logit
matriz_confusion

# Cargar datos reales del INEC.
data_inec <- suppressMessages(read_excel("data/data_pobreza_INEC.xlsx"))

# 1. Desde una celda R con system()
if (nzchar(Sys.which("conda"))) {
  system("conda run -n rbase jupyter nbconvert --to html modulo1.ipynb")
} else {
  message("No se ejecutó nbconvert porque conda no está disponible en PATH.")
}
