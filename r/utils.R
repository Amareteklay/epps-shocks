suppressPackageStartupMessages({
  library(lme4)
  library(glmmTMB)
  library(broom.mixed)
  library(dplyr)
  library(purrr)
  library(readr)
})

safe_AICc <- function(fit){
  k <- length(fixef(fit))  # for glmmTMB or glmer, adjust if needed
  n <- nobs(fit)
  AIC(fit) + (2*k*(k+1))/(n-k-1)
}

fit_one <- function(df, formula_str, family = binomial(), engine = c("glmmTMB","glmer")){
  engine <- match.arg(engine)
  fm <- as.formula(formula_str)
  res <- tryCatch({
    fit <- if (engine == "glmmTMB") {
      glmmTMB(fm, data = df, family = family)
    } else {
      glmer(fm, data = df, family = family,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
    }
    tibble(
      aic = AIC(fit),
      aicc = safe_AICc(fit),
      logLik = as.numeric(logLik(fit)),
      converged = isTRUE(summary(fit)$optinfo$conv$opt == 0) | isTRUE(is.null(summary(fit)$optinfo$conv$lme4$messages)),
      tidy = list(broom.mixed::tidy(fit, effects = "fixed"))
    )
  }, error = function(e) tibble(aic = NA_real_, aicc = NA_real_, logLik = NA_real_, converged = FALSE, tidy = list(tibble())))
  res
}
