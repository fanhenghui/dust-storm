const express = require('express');
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
    let name = req.body.uname;
    let pwd = req.body.upwd;

    req.db.getConnection((err, connection) =>{
        if (err) {
            //done with the connection
            connection.release();
            //database error
            console.log(`DB connect error when sign in: ${err}`);
            res.status(500).send('数据库操作失败');
        } else {
            connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, data, fields)=>{
                if(err) {
                    //done with the connection
                    connection.release();
                    //database error
                    console.log(`DB error when sign in: ${err}`);
                    res.status(500).send('数据库操作失败');
                } else if(data.length == 0) {
                    //done with the connection
                    connection.release();
                    //user doesn't exist
                    req.session.error = '用户名或者密码错误';
                    res.status(404).send('用户名或者密码错误');
                } else {
                    if (data[0].password != pwd ) {
                        //done with the connection
                        connection.release();
                        //error password
                        req.session.error = '用户名或者密码错误';
                        res.status(404).send('用户名或者密码错误');
                    } else if (data[0].online[0] === 1) {
                        //done with the connection
                        connection.release();
                        //repeat login
                        req.session.error = '用户已经在线，不可以重复登录';
                        res.status(404).send('用户已经在线，不可以重复登录');
                    } else {
                        //update online 
                        connection.query(`UPDATE usr SET online=1 WHERE name='${name}'`, (err, data, fields)=> {
                            //done with the connection
                            connection.release();
                            if (err) {
                                //database error
                                console.log(`DB error when sign in: ${err}`);
                                res.status(500).send('数据库操作失败');
                            } else {
                                //login successs
                                req.session.user = {name:name};
                                res.sendStatus(200);
                            }
                        });
                        
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
    let name = req.body.uname;
    let pwd = req.body.upwd;

    req.db.getConnection((err, connection) =>{
        if (err) {
            //done with the connection
            connection.release();
            //database error
            console.log(`DB connect error when register: ${err}`);
            res.status(500).send('数据库操作失败');
        } else {
            connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, data, fields)=>{
                if(err) {
                    //done with the connection
                    connection.release();
                    //database error
                    console.log(`DB connect error when register: ${err}`);
                    res.status(500).send('数据库操作失败');
                } else if (data.length != 0) {
                    //done with the connection
                    connection.release();
                    //username has exist
                    req.session.error = '用户名已存在';
                    res.status(500).send('用户名已存在');
                } else {
                    connection.query(`INSERT INTO usr(name,role,online,password) VALUES ('${name}',0,0,'${pwd}')`, (err, data, fields)=>{
                        //done with the connection
                        connection.release();
                        if(err) {
                            //database error
                            console.log(`DB connect error when register: ${err}`);
                            res.status(500).send('数据库操作失败');
                        } else {
                            //register success
                            req.session.user = {name:name};//record to session
                            res.send(200);
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
    req.session.error = null;
    res.redirect('/user/login');
});