-- create user
create user postgres password 'postgres12345';
grant all on database mlops_db to postgres;
grant pg_read_server_files to postgres;
grant pg_write_server_files to postgres;