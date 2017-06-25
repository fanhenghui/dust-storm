express = require("express");
util = require("util");
var bodyparser = require("body-parser");
var app = express();

var urlencodedParser =bodyparser.urlencoded({extended:false});

app.use(express.static("public"));

app.get("/index.html" , function(req , res){
    console.log("hostname : " + req.hostname);
    console.log("ip : " + req.ip);
    console.log("orignal URL : " + req.orignalUrl);
    console.log("base URL : " + req.baseUrl);
    console.log("path : " + req.path);
    console.log("protocol : " + req.protocol);
    console.log("query : " + util.inspect(req.query));
    console.log("route : " + util.inspect(req.route));
    console.log("subdomains : " + req.subdomains);
    console.log("body : " + req.body);

    //res.send("hello world");
    res.sendFile(__dirname + "/" +"index.html");

});

app.get("/process_get" , function(req , res){

    var response = { 
        "firstname" : req.query.first_name ,
        "lastname" : req.query.last_name 
    };
    console.log(util.inspect(response));
    res.sendFile(__dirname + "/" + "jump.html");
    //res.end(JSON.stringify(response));
});

app.post("/process_get" , urlencodedParser , function(req , res){

    // var response = { 
    //     "firstname2" : req.query.first_name ,
    //     "lastname2" : req.query.last_name 
    // };
    // console.log(util.inspect(response));
    // res.end(JSON.stringify(response));

    var response = {
       "first_name":req.body.first_name,
       "last_name":req.body.last_name
   };
   console.log(response);
   res.end(JSON.stringify(response));
});


app.get("/ok" , function(req , res){

    res.send("OK");
});


app.get("/error" , function(req , res){

    res.send("error");
});

var server  = app.listen(8080 , function(){

    var host = server.address().address;
    var port = server.address().port;

    console.log("address is http://%s:%s" , host , port);
});