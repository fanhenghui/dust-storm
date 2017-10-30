var express = require('express');
var router = express.Router();

router.get('/', function (req, res, next) {
  // useless code cookies 已经以session的形式发送了
  // if (req.cookies.isVisit) {
  //   console.log('welcome back.');
  //   console.log(req.cookies);
  // } else {
  //   let cookieName = require('./server-ip.js').serverip + new Date().getTime();
  //   res.cookie('isVisit', 1, {maxAge: 60 * 1000});
  //   res.cookie('name', cookieName);
  //   console.log("welcome.");
  // }
  res.render('index', {
    title: 'Web-based Medical Viewer from Baidu'
  }); // 到达此路径则渲染index文件，并传出title值供 index.html使用
});
router.route('/login').get(function (req, res) { // 到达此路径则渲染login文件，并传出title值供 login.html使用
  res.render('login', {
    title: 'User Login'
  });
}).post(function (req, res) { // 从此路径检测到post方式则进行post数据的处理操作
  //这里的User就是从model中获取user对象，通过global.dbHandel全局方法
  var User = global.dbHandel.getModel('user');
  var uname = req.body.uname; //获取post上来的 data数据中 uname的值
  console.log(uname);
  // alreadyLoggedUsers.includes(uname)
  if (global.userList.includesUser(uname)) {
    req.session.error = '已经登录';
    res.status(404).send('已经登录');
    // res.send(500);
  } else {
    User.findOne({
      name: uname
    }, function (err, doc) { //通过此model以用户名的条件查询数据库中的匹配信息
      if (err) { //错误就返回给原post处（login.html) 状态码为500的错误
        res.send(500);
        console.log(err);
      } else if (!doc) { //查询不到用户名匹配信息，则用户名不存在
        req.session.error = '用户名不存在';
        res.status(404).send('用户名不存在');//	状态码返回404
      } else {
        if (req.body.upwd != doc.password) { //查询到匹配用户名的信息，但相应的password属性不匹配
          req.session.error = '密码错误';
          res.status(404).send('密码错误');
        } else { //信息匹配成功，则将此对象（匹配到的user) 赋给session.user 并返回成功
          // alreadyLoggedUsers.push(doc.name);
          global.userList.addLoggedUser(uname)
          req.session.user = doc;
          res.send(200);
        }
      }
    });
  }
});

router.route('/register').get(function(req, res) { 
  res.render('register', {title: 'User register'});
}).post(function(req, res) {
    //这里的User就是从model中获取user对象，通过global.dbHandel全局方法
    var User = global.dbHandel.getModel('user');
    var uname = req.body.uname;
    var upwd = req.body.upwd;
    User.findOne( {name: uname}, function(err, doc) {  // 同理 /login 路径的处理方式
      if (err) {
        res.send(500);
        req.session.error = '网络异常错误！';
        console.log(err);
      } else if (doc) {
        req.session.error = '用户名已存在！';
        res.send(500);
      } else { // 创建一组user对象置入model
        User.create( { name: uname, password: upwd }, function(err, doc) {
          if (err) {
            res.send(500);
            console.log(err);
          } else {
            req.session.error = '用户名创建成功！';
            res.send(200);
          }
        });
      }
    });
  });

router.get('/home', function(req, res) {
  if (!req.session.user) {  //到达/home路径首先判断是否已经登录
    req.session.error = '请先登录'
    res.redirect('/login');  //未登录则重定向到 /login 路径
  }
  res.render(
      'home',
      {title: 'Home', name: req.session.user.name});  //已登录则渲染home页面
});

router.get('/logout', function(req, res) {  // 到达 /logout 路径则登出
  // session中user,error对象置空，并重定向到根路径
  req.session.user = null;
  req.session.error = null;
  res.redirect('/');
});

router.route('/review').get(function(req, res) {
  if (!req.session.user) {  //到达/home路径首先判断是否已经登录
    req.session.error = '请先登录'
    res.redirect('/login');  //未登录则重定向到 /login 路径
    }
  else {
    console.log('login name is : ', req.session.user.name);
    var ip = require('./server-ip.js').serverip;
    //send user name and server ip
    res.render('review',{username: req.session.user.name, serverip: ip});  // go directly to review page
  }
});
module.exports = router;
