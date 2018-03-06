const express = require('express');
const common = require('../libs/common');
const config = require('../config/config');

let router = express.Router();

module.exports = router;

router.get('/', (req, res, next)=> {
    res.redirect('/user/login');
});

router.route('/login')
.get((req, res)=>{
    //这里可以给出服务器的UID
    res.render('login', {title: 'User Login'});``
})
.post((req, res)=>{
    const {name, pass, verify} = req.body;

    //TODO 校验用户名密码合法性

    req.db.getConnection((err, connection) =>{
        if (err) {
            //done with the connection
            connection.release();
            //database error
            console.log(`DB connect error when sign in: ${err}`);
            res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
        } else {
            connection.query(`SELECT online_token,password FROM user WHERE name='${name}'`, (err, data, fields)=>{
                if(err) {
                    //done with the connection
                    connection.release();
                    //database error
                    console.log(`DB error when sign in: ${err}`);
                    res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
                } else if(data.length == 0) {
                    //done with the connection
                    connection.release();
                    //user doesn't exist
                    res.send({err:-1, msg:'用户名或者密码错误'});
                } else {
                    if (data[0].password != common.md5(pass) ) {
                        //done with the connection
                        connection.release();
                        //error password
                        res.send({err:-1, msg:'用户名或者密码错误'});
                    } else {
                        if (!data[0].online_token || verify == 1) {
                            const new_token = common.uuid();
                            //update online 
                            connection.query(`UPDATE user SET online_token='${new_token}' WHERE name='${name}'`, (err, data, fields)=> {
                               //done with the connection
                               connection.release();
                               if (err) {
                                   //database error
                                   console.log(`DB error when sign in: ${err}`);
                                   res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
                               } else {
                                   //login successs
                                   req.session.user = {name:name};
                                   res.cookie('online_token', new_token);
                                   res.send({err:0, msg:'登录成功'});
                               }
                            });
                        } else {
                            //ask user to verify
                            res.send({err:1, msg:'账号已经在其他页面登录'});
                        }
                    }
                }
            });
        }
    });
});

router.route('/register')
.get((req, res)=>{
    res.render('register', {
        title: 'User register'
    });
})
.post((req, res)=>{
    const {name, pass} = req.body;

    //TODO 校验用户名密码合法性

    let passHash = common.md5(pass);
    req.db.getConnection((err, connection) =>{
        if (err) {
            //done with the connection
            connection.release();
            //database error
            console.log(`DB connect error when register: ${err}`);
            res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
        } else {
            connection.query(`SELECT ID FROM user WHERE name='${name}'`, (err, data, fields)=>{
                if(err) {
                    //done with the connection
                    connection.release();
                    //database error
                    console.log(`DB connect error when register: ${err}`);
                    res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
                } else if (data.length != 0) {
                    //done with the connection
                    connection.release();
                    //username has exist
                    res.send({err:-1, msg:'用户名已存在'});
                } else {
                    const id = common.uuid();
                    connection.query(`INSERT INTO user(ID,name,role,password) VALUES ('${id}','${name}',0,'${passHash}')`, (err, data, fields)=>{
                        //done with the connection
                        connection.release();
                        if(err) {
                            //database error
                            console.log(`DB connect error when register: ${err}`);
                            res.send({err:-1, msg:'数据库操作失败,请稍后再试'});
                        } else {
                            //register success
                            req.session.user = {name:name};//record to session
                            res.send({err:0, msg:'注册成功'});
                        }
                    });
                }
            });
        }
    });
});

router.get('/logout', (req, res)=>{
    // clear session's user/error and redirect to login
    req.session.user = null;
    res.redirect('/user/login');
});