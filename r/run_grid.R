# r/run_grid.R — supports BOTH grid schemas (new: spec_id/scope/dv/rhs/engine, old: ModelID/Scope/Predictors/FixedEffects/OutbreakType)
# Usage:
#   Rscript r/run_grid.R specs/model_grid_0001.csv data/03_processed/full_panel.csv results/lags

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(purrr); library(tidyr)
  library(lme4);  library(glmmTMB); library(broom.mixed)
  library(stringr); library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || (is.character(a) && (length(a)==0 || is.na(a)))) b else a

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript r/run_grid.R <grid_csv> <panel_csv> <out_dir>")
grid_path  <- args[[1]]
panel_path <- args[[2]]
out_dir    <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------- helpers ----------
parse_predictors <- function(x) {
  if (is.null(x) || is.na(x) || x == "") return(character())
  # try JSON first
  pr <- tryCatch(fromJSON(x), error = function(e) NULL)
  if (is.character(pr)) return(pr)
  # fallback: strip brackets/quotes and split
  x2 <- gsub("^\\[|\\]$", "", x)
  x2 <- gsub("['\"]", "", x2)
  parts <- unlist(strsplit(x2, "\\s*,\\s*"))
  parts[nzchar(parts)]
}

build_formula_legacy <- function(dv, predictors, fixed_effects, scope) {
  pred_str <- if (length(predictors)) paste(predictors, collapse = " + ") else "1"
  fe_str <- fixed_effects %||% ""
  if (identical(scope, "Global")) {
    if (!grepl("\\(1\\|Continent\\)", fe_str)) fe_str <- paste(fe_str, "+ (1|Continent)")
    if (!grepl("\\(1\\|Country\\)", fe_str))   fe_str <- paste(fe_str, "+ (1|Country)")
  } else {
    if (!grepl("\\(1\\|Country\\)", fe_str))   fe_str <- paste(fe_str, "+ (1|Country)")
  }
  fe_str <- gsub("\\+\\s*\\+", "+", fe_str)
  rhs <- trimws(paste(pred_str, fe_str))
  sprintf("%s ~ %s", dv, rhs)
}

safe_fit <- function(df, formula_str, engine=c("glmmTMB","glmer")) {
  engine <- match.arg(engine)
  fm <- as.formula(formula_str)
  tryCatch({
    fit <- if (engine=="glmmTMB") {
      glmmTMB(fm, data=df, family=binomial())
    } else {
      glmer(fm, data=df, family=binomial(),
            control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
    }
    tibble(
      ok=TRUE,
      aic=AIC(fit),
      bic=BIC(fit),
      aicc=NA_real_,
      logLik=as.numeric(logLik(fit)),
      converged=TRUE,
      tidy=list(broom.mixed::tidy(fit, effects="fixed"))
    )
  }, error=function(e){
    tibble(ok=FALSE, aic=NA_real_, bic=NA_real_, aicc=NA_real_,
           logLik=NA_real_, converged=FALSE, error=conditionMessage(e),
           tidy=list(tibble(term=character(), estimate=double(), std.error=double(), statistic=double(), p.value=double())))
  })
}

# ---------- load ----------
grid <- readr::read_csv(grid_path, show_col_types = FALSE)
df   <- readr::read_csv(panel_path, show_col_types = FALSE)

# types
if (!"Year" %in% names(df)) stop("Panel missing 'Year'")
df$Year <- as.integer(df$Year)

# resolve continent labels for diagnostics
continents <- sort(unique(df$Continent))
# message("Detected Continent labels: ", paste(continents, collapse=", "))

# detect schema + accessors
has_new <- all(c("scope") %in% tolower(names(grid))) && ("rhs" %in% tolower(names(grid)))
cols_l <- tolower(names(grid))
getcol <- function(nm) {
  # case-insensitive access
  idx <- which(cols_l == tolower(nm))
  if (length(idx)) grid[[idx[1]]] else NULL
}

rows_list <- split(grid, seq_len(nrow(grid)))
out_rows <- vector("list", length(rows_list))

for (i in seq_along(rows_list)) {
  r <- rows_list[[i]][1, , drop=FALSE]

  if (has_new) {
    # NEW SCHEMA (spec_id/scope/dv/rhs/engine)
    scope   <- getcol("scope")[[i]]  %||% "Global"
    dv      <- getcol("dv")[[i]]     %||% "outbreak"
    rhs     <- getcol("rhs")[[i]]    %||% ""
    engine  <- getcol("engine")[[i]] %||% "glmmTMB"
    spec_id <- getcol("spec_id")[[i]] %||% paste0("spec_", i)

    if (!identical(scope, "Global")) {
      if (!"Continent" %in% names(df)) {
        out_rows[[i]] <- tibble(spec_id=spec_id, scope=scope, dv=dv, formula=NA_character_,
                                n=NA_integer_, engine=engine, aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                                logLik=NA_real_, converged=FALSE,
                                error="Panel missing 'Continent' column")
        next
      }
      if (!(scope %in% df$Continent)) {
        out_rows[[i]] <- tibble(spec_id=spec_id, scope=scope, dv=dv, formula=NA_character_,
                                n=0L, engine=engine, aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                                logLik=NA_real_, converged=FALSE,
                                error=paste0("Scope '", scope,"' not present in Continent"))
        next
      }
      sub <- dplyr::filter(df, .data$Continent == scope)
    } else {
      sub <- df
    }

    if (!dv %in% names(sub) || dplyr::n_distinct(sub[[dv]], na.rm=TRUE) < 2) {
      out_rows[[i]] <- tibble(spec_id=spec_id, scope=scope, dv=dv, formula=NA_character_,
                              n=nrow(sub), engine=engine, aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                              logLik=NA_real_, converged=FALSE,
                              error=paste0("DV '", dv, "' missing or single class in scope"))
      next
    }

    rhs_trim <- trimws(rhs)
    if (is.na(rhs_trim) || rhs_trim == "" || rhs_trim == "~" || rhs_trim == "1") {
      # ensure at least Year + random intercept
      rhs_trim <- "scale(Year) + (1|Country)"
    }
    # build full formula from rhs (already RHS)
    form <- sprintf("%s ~ %s", dv, rhs_trim)

    # limit columns to used
    used <- unique(c(all.vars(as.formula(form)), "Country","Continent"))
    used <- intersect(used, names(sub))
    sub2 <- tidyr::drop_na(sub[, used, drop=FALSE])

    res <- safe_fit(sub2, form, engine = ifelse(engine %in% c("glmer","glmmTMB"), engine, "glmmTMB"))

    out_rows[[i]] <- tibble(
      spec_id = spec_id, scope = scope, dv = dv, formula = form,
      n = nrow(sub2), engine = engine,
      aic = res$aic, aicc = res$aicc, bic = res$bic, logLik = res$logLik,
      converged = res$converged,
      error = if ("error" %in% names(res)) res$error else NA_character_
    )

  } else {
    # LEGACY SCHEMA (ModelID/Scope/FixedEffects/Predictors/OutbreakType)
    scope <- (r$Scope %||% r$scope %||% "Global")[[1]]
    fe    <- (r$FixedEffects %||% r$fixedeffects %||% "")[[1]]
    preds <- parse_predictors((r$Predictors %||% r$predictors %||% "")[[1]])
    dv    <- (r$OutbreakType %||% r$outbreaktype %||% "outbreak")[[1]]
    model_id <- (r$ModelID %||% r$modelid %||% paste0("mod_", i))[[1]]

    if (!identical(scope, "Global")) {
      if (!"Continent" %in% names(df)) {
        out_rows[[i]] <- tibble(ModelID=model_id, Scope=scope, dv=dv, formula=NA_character_,
                                n=NA_integer_, engine="glmer", aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                                logLik=NA_real_, converged=FALSE, error="Panel missing 'Continent' column")
        next
      }
      if (!(scope %in% df$Continent)) {
        out_rows[[i]] <- tibble(ModelID=model_id, Scope=scope, dv=dv, formula=NA_character_,
                                n=0L, engine="glmer", aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                                logLik=NA_real_, converged=FALSE,
                                error=paste0("Scope '", scope,"' not present in Continent"))
        next
      }
      sub <- dplyr::filter(df, .data$Continent == scope)
    } else {
      sub <- df
    }

    # keep only predictors that exist & vary
    preds <- preds[preds %in% names(sub)]
    if (length(preds)) {
      vary <- vapply(preds, function(nm) dplyr::n_distinct(sub[[nm]], na.rm=TRUE) > 1, logical(1))
      preds <- preds[vary]
    }

    if (!dv %in% names(sub) || dplyr::n_distinct(sub[[dv]], na.rm=TRUE) < 2) {
      out_rows[[i]] <- tibble(ModelID=model_id, Scope=scope, dv=dv, formula=NA_character_,
                              n=nrow(sub), engine="glmer", aic=NA_real_, aicc=NA_real_, bic=NA_real_,
                              logLik = NA_real_, converged=FALSE,
                              error=paste0("DV '", dv, "' missing or single class in scope"))
      next
    }

    used_cols <- unique(c(preds, dv, "Country","Continent","Year"))
    sub2 <- tidyr::drop_na(sub[, intersect(used_cols, names(sub)), drop=FALSE])

    form <- build_formula_legacy(dv, preds, fe, scope)
    res  <- safe_fit(sub2, form, engine="glmer")

    out_rows[[i]] <- tibble(
      ModelID = model_id, Scope = scope, dv = dv, formula = form,
      n = nrow(sub2), engine = "glmer",
      aic = res$aic, aicc = res$aicc, bic = res$bic, logLik = res$logLik,
      converged = res$converged,
      error = if ("error" %in% names(res)) res$error else NA_character_,
      Predictors = list(preds),
      FixedEffectsSpec = fe
    )
  }
}

results <- dplyr::bind_rows(out_rows)

# write model-level
base <- tools::file_path_sans_ext(basename(grid_path))
out_main <- file.path(out_dir, paste0("batch_", base, ".csv"))
readr::write_csv(results, out_main)
