const mysql = require('mysql');
const fs = require('fs');
const config = require('../config/config')

module.exports = {
    createDB(gContainer) {
        fs.readFile(config.app_config_path, (err,data)=>{
            let configStr = data.toString();
            let configLines = configStr.split('\n');
            if (configLines.length < 5) {
                console.log('config file damadged.')
                return {};
            }
            let addr = {};
            for (let i = 0; i< configLines.length; ++i) {
                let items = configLines[i].split(' ');
                if (items.length == 3 && items[0] == 'DBIP')	 {
                    addr['ip'] = items[2];
                } else if (items.length == 3 && items[0] == 'DBPort')	 {
                    addr['port'] = items[2];
                } else if (items.length == 3 && items[0] == 'DBUser')	 {
                    addr['user'] = items[2];
                } else if (items.length == 3 && items[0] == 'DBPWD')	 {
                    addr['password'] = items[2];
                } else if (items.length == 3 && items[0] == 'DBName')	 {
                    addr['database'] = items[2];
                }
        
                if (i > 6 && addr['ip'] != undefined &&
                    addr['port'] != undefined &&
                    addr['user'] != undefined &&
                    addr['password'] != undefined &&
                    addr['database'] != undefined) {
                    break;
                }
            }
    
            let db = mysql.createPool( {
                connectionLimit:10,//TODO，这里的值最好要和server带的用户数量匹配，估计N/2就够了
                host:addr.ip,
                port:addr.port,
                user:addr.user,
                password:addr.password,
                database:addr.database,
            });
            
    
            db.on('release', function (connection) {
                console.log('Connection %d released', connection.threadId);
            });

            gContainer.db = db;
        });
    },

    releaseDB(db) {
		console.log('release DB');
		db.end(err=>{
			if(err) {
				console.log(`DB error when release DB pool: ${err}`);
			}
		});
	},

    signOut: function(db, name) {
		if (db == null) {
            console.log('DB instance is null.')
			return;
		}
		db.getConnection((err, connection) =>{
			if (err) {
				console.log(`DB connect error when sign in: ${err}`);
			} else {
				connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
					if(err) {
						//done with the connection
						connection.release();
						console.log(`DB error when sign out: ${err}`);
					} else {
						connection.query(`UPDATE usr SET online=0 WHERE name='${name}'`, (err, res, fields)=> {
							//done with the connection
							connection.release();
							if (err) {
								//database error
								console.log(`DB error when sign out: ${err}`);
							}
						});
					}
				});
			}
		});
    }
}