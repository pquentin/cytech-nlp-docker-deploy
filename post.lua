-- example HTTP POST script which demonstrates setting the
-- HTTP method, body, and adding a header

wrk.method = "POST"
wrk.body   = '{"text":"This was the biggest hit movie of 1971"}'
wrk.headers["Content-Type"] = "application/json"
