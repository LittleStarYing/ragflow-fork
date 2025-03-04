curl 'http://localhost:9222/v1/document/run' \
  -H 'Accept: application/json' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'Authorization: ImIzN2M2ZWY2ZjhjYjExZWY5YTA4YmZkNmE3NTQ1MjE1Ig.Z8auBA.srtIDUzS9BZPVh-y-QDqsDTZ-_I' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json;charset=UTF-8' \
  -H 'Cookie: x-hng=lang=zh-CN&domain=localhost; username-localhost-8888=2|1:0|10:1740122703|23:username-localhost-8888|188:eyJ1c2VybmFtZSI6ICI1NTlmZTk0NmYwNzk0NzY4ODBlMDQ0MWM4MWZmN2M2YiIsICJuYW1lIjogIkFub255bW91cyBMZWRhIiwgImRpc3BsYXlfbmFtZSI6ICJBbm9ueW1vdXMgTGVkYSIsICJpbml0aWFscyI6ICJBTCIsICJjb2xvciI6IG51bGx9|46da06ed3cd95e6e7c9c65776d08fd72c9bd61b546aab755df3b78a33039b596; _xsrf=2|a74cecf5|3d5142dd3d4518b34f0ccf28494c13c6|1740122703; session=mnnTHVbwmEBVhidDAtRGU568bIM8JsT7To6ppAxm1vc' \
  -H 'Origin: http://localhost:9222' \
  -H 'Referer: http://localhost:9222/knowledge/dataset?id=f36f2d44f83c11efa5ad0242ac120006' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  --data-raw '{"doc_ids":["fb3a6ed0f83c11ef850a0242ac120006"],"run":1,"delete":false}'






curl 'http://localhost:9222/v1/chunk/list' \
  -H 'Accept: application/json' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'Authorization: ImIzN2M2ZWY2ZjhjYjExZWY5YTA4YmZkNmE3NTQ1MjE1Ig.Z8auBA.srtIDUzS9BZPVh-y-QDqsDTZ-_I' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json;charset=UTF-8' \
  -H 'Cookie: x-hng=lang=zh-CN&domain=localhost; username-localhost-8888=2|1:0|10:1740122703|23:username-localhost-8888|188:eyJ1c2VybmFtZSI6ICI1NTlmZTk0NmYwNzk0NzY4ODBlMDQ0MWM4MWZmN2M2YiIsICJuYW1lIjogIkFub255bW91cyBMZWRhIiwgImRpc3BsYXlfbmFtZSI6ICJBbm9ueW1vdXMgTGVkYSIsICJpbml0aWFscyI6ICJBTCIsICJjb2xvciI6IG51bGx9|46da06ed3cd95e6e7c9c65776d08fd72c9bd61b546aab755df3b78a33039b596; _xsrf=2|a74cecf5|3d5142dd3d4518b34f0ccf28494c13c6|1740122703; session=mnnTHVbwmEBVhidDAtRGU568bIM8JsT7To6ppAxm1vc' \
  -H 'Origin: http://localhost:9222' \
  -H 'Referer: http://localhost:9222/knowledge/dataset/chunk?id=0d2b5454f8d511efb860f75bc463159f&doc_id=a112e146f8d511efb860f75bc463159f' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  --data-raw '{"doc_id":"a112e146f8d511efb860f75bc463159f","page":1,"size":10,"keywords":""}'