
var fs = require('fs');

//阻塞调用
// var data = fs.readFileSync('input.txt');

// console.log(data.toString());
// console.log('程序执行结束');

//回调函数实现异步编程（非阻塞）
fs.readFile( 'input.txt' , function(err ,data){
    if(err) return console.error(err);
    console.log(data.toString());
})

console.log('程序执行结束');