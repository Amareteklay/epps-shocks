source("r/utils.R")

fit_spec <- function(df, spec_row){
  # spec_row carries: scope, dv, predictors, random, year_term, id/hash, etc.
  scope_df <- if (spec_row$scope == "Global") df else dplyr::filter(df, Continent == spec_row$scope)
  rhs <- paste(c(spec_row$predictors, spec_row$year_term, spec_row$random), collapse = " + ")
  formula_str <- sprintf("%s ~ %s", spec_row$dv, rhs)
  out <- fit_one(scope_df, formula_str, family = binomial(), engine = spec_row$engine)
  tibble::tibble(
    spec_id = spec_row$spec_id,
    scope   = spec_row$scope,
    dv      = spec_row$dv,
    formula = formula_str,
    n       = nrow(scope_df)
  ) |>
    dplyr::bind_cols(out)
}
