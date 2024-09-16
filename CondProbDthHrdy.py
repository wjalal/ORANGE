def cond_prob_dthhrdy_gt(df_agegap, i, U, silent=False):
    denominator = len(df_agegap.query(f"agegap>={U}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap>={U}")) / denominator
    if not silent:
        print(f"p(dthhrdy={i} | agegap>={U:.4f}) = {p:.4f}")
    return p

def cond_prob_dthhrdy_lt(df_agegap, i, L, silent=False):
    denominator = len(df_agegap.query(f"agegap<={L}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap<={L}")) / denominator
    if not silent:
        print(f"p(dthhrdy={i} | agegap<={L:.4f}) = {p:.4f}")
    return p

def cond_prob_dthhrdy_range(df_agegap, i, L, U, silent=False):
    denominator = len(df_agegap.query(f"agegap>={L} and agegap<={U}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap>={L} and agegap<={U}")) / denominator
    if not silent:
        print(f"p(dthhrdy={i} | {L:.4f}<agegap<{U:.4f}) = {p:.4f}")
    return p

def prob_dthhrdy(df_agegap, i, silent=False):
    denominator = df_agegap.shape[0]
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy == {i}")) / denominator
    if not silent:
        print(f"p(dthhrdy={i}) = {p:.4f}")
    return p

def cond_prob_gt_dthhrdy(df_agegap, U, i, silent=False):
    denominator = len(df_agegap.query(f"dthhrdy=={i}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap>={U}")) / denominator
    if not silent:
        print(f"p(agegap>={U:.4f} | dthhrdy={i}) = {p:.4f}")
    return p

def cond_prob_lt_dthhrdy(df_agegap, L, i, silent=False):
    denominator = len(df_agegap.query(f"dthhrdy=={i}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap<={L}")) / denominator
    if not silent:
        print(f"p(agegap<={L:.4f} | dthhrdy={i}) = {p:.4f}")
    return p

def cond_prob_range_dthhrdy(df_agegap, L, U, i, silent=False):
    denominator = len(df_agegap.query(f"dthhrdy=={i}"))
    if denominator == 0:
        p = 0
    else:
        p = len(df_agegap.query(f"dthhrdy=={i} and agegap>={L} and agegap<={U}")) / denominator
    if not silent:
        print(f"p({L:.4f}<agegap<{U:.4f} | dthhrdy={i}) = {p:.4f}")
    return p
