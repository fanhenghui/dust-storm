// var http = require('http');
// var fs = require('fs');
// var url = require('url');

// http.createServer( function (request , response){
//     //解析域名，包括文件名
//     var pathname = url.parse(request.url).pathname;

//     //输出请求的文件名字 
//     console.log('request for ' + pathname + ' received.');

//     //从文件系统中读取请求的文件内容
//     fs.readFile(pathname.substr(1) , function (err , data){
//         if(err){
//             console.log(err);
//             //HTTP 状态码 : 404 NOT FOUND
//             //Content Type : text/plain
//             response.writeHead(404 , {'Content-Type': 'text/html'});
//         }else {
//             //HTTP 状态码 ： 200 OK
//             //content type : text/plain
//             response.writeHead(200 , {'Content-Type' : 'text/html'});
//             response.write(data.toString());
//         }
//         //发送相应数据
//         response.end();
//     });
// }).listen(8081);

// console.log('Server running at http://127.0.0.1:8081/');


var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var cp = require('child_process')
 
// 创建 application/x-www-form-urlencoded 编码解析
var urlencodedParser = bodyParser.urlencoded({ extended: false })
 
app.use(express.static('public'));
 
app.get('/index.html', function (req, res) {
   res.sendFile( __dirname + "/" + "index.html" );
})
 
app.post('/process_post', urlencodedParser, function (req, res) {
 
   // 输出 JSON 格式
   var response = {
       "first_name":req.body.first_name,
       "last_name":req.body.last_name
   };
   console.log(response);
   res.end(JSON.stringify(response));
})
 
var server = app.listen(8081, function () {
  var host = server.address().address
  var port = server.address().port
 
  console.log("应用实例，访问地址为 http://%s:%s", host, port)
 
})