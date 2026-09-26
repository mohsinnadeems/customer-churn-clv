from churnclv.data import clean, orders


def test_clean_rules(raw):
    tx = clean(raw.rename(columns={"Customer ID": "CustomerID"}))
    assert "POST" not in set(tx["StockCode"])
    assert tx["CustomerID"].notna().all() and (tx["Price"] > 0).all()
    assert len(tx) == 4  # two lines, one return, one de-duplicated line
    ret = tx[tx["is_return"]]
    assert len(ret) == 1 and ret["Revenue"].iat[0] == -5.0


def test_orders_exclude_returns(raw):
    tx = clean(raw.rename(columns={"Customer ID": "CustomerID"}))
    o = orders(tx)
    assert set(o["Invoice"]) == {"1001", "1005"}
    assert o.loc[o["Invoice"] == "1001", "revenue"].iat[0] == 20.0
